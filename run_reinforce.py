from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import logging
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score
from ast import literal_eval as make_tuple
import copy

from transformers import (RobertaTokenizer, AlbertTokenizer, BertTokenizer)

from utils_embedding_model import feature_loader, load_features, save_features
from utils_real_data import example_loader
from utils_reinforce import train_classifier_discriminator, train_generator, word_index_selector, my_CrossEntropyLoss
from utils_model_maitenence import save_models, load_models, inititalize_models
from utils_ablation import ablation, ablation_discriminator, discriminator_eval
from utils_generator import Weight_Clipper
from utils_eval import write_out_discriminator_eval, stream_data

# logging
logger = logging.getLogger(__name__)

# tanh and sigmoid
tanh = nn.Tanh()

# weight clipper
weight_clipper = Weight_Clipper()

# hugging face transformers default models, can use pretrained ones though too
TOKENIZER_CLASSES = {
    'roberta': RobertaTokenizer,
    'albert': AlbertTokenizer,
    'bert': BertTokenizer,
}

def set_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


# from hf, returns a list of lists from features of a selected field within the choices_features list of dicts
def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def detach_inputs(tuple_of_tensors):

    out_tensors = (t.detach() if hasattr(t, 'detach') else t for t in tuple_of_tensors)

    return out_tensors


# return if there is a gpu available
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device('cuda:6')
#     else:
#         return torch.device('cpu')


# returns a list of length num_choices with each entry a length four list with a 1 in the correct response spot and 0s elsewhere
def label_map(labels, num_choices):

    def label_list(target, new_target, switch_index, num_choices):
        l = [target]*num_choices
        l[switch_index] = new_target
        return l

    answers = [label_list(0, 1, lab, num_choices) for lab in labels]
    return answers


def flip_labels(classification_labels=None, discriminator_labels=None):

    out_c_labels = None
    out_d_labels = None

    if classification_labels is not None:
        out_c_labels = 1 - classification_labels
    if discriminator_labels is not None:
        out_d_labels = discriminator_labels * -1

    return out_c_labels, out_d_labels


def load_and_cache_features(args, tokenizer, subset):
    assert subset in ['train', 'dev', 'test']

    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)
    corpus_str = '_WithCorpus' if (args.use_corpus and subset == 'train') else ''
    cached_features_filename = os.path.join(args.cache_dir, '{}_{}_{}{}{}{}'.format(subset,
                                                                                    args.tokenizer_name,
                                                                                    args.max_length,
                                                                                    '_'+'-'.join(args.domain_words),
                                                                                    cutoff_str,
                                                                                    corpus_str))
    if os.path.exists(cached_features_filename) and not args.overwrite_cache_dir:
        logger.info('Loading features from ({})'.format(cached_features_filename))
        features = load_features(cached_features_filename)
    else:
        # load in examples and features for training
        logger.info('Creating examples and features from ({})'.format(args.data_dir))
        examples = example_loader(args, subset=subset)
        features = feature_loader(args, tokenizer, examples)

        assert save_features(features, cached_features_filename) == -1

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_token_type_mask = torch.tensor(select_field(features, 'token_type_mask'), dtype=torch.long)
    all_attention_mask = torch.tensor(select_field(features, 'attention_mask'), dtype=torch.float)
    all_classification_labels = torch.tensor(label_map([f.classification_label for f in features], num_choices=4), dtype=torch.float)
    all_sentences_types = torch.tensor([f.sentences_type for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_mask, all_attention_mask, all_classification_labels, all_sentences_types)

    return dataset


def train(args, tokenizer, dataset, generatorM, attentionM, classifierM, discriminatorM):

    # use pytorch data loaders to cycle through the data
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    # optimizers
    classifierO = AdamW(classifierM.parameters(), lr=args.learning_rate_classifier, weight_decay=1)
    generatorO = AdamW(generatorM.parameters(), lr=args.learning_rate_generator, weight_decay=0)
    discriminatorO = AdamW(discriminatorM.parameters(), lr=args.learning_rate_discriminator, weight_decay=1)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    update_step = 0
    global_step = 0

    # zero out gradient of networks
    generatorM.zero_grad()
    classifierM.zero_grad()
    discriminatorM.zero_grad()

    if args.do_prevaluation:
        assert -1 == evaluate(args, classifierM, generatorM, attentionM, tokenizer, 0)

    args.optimization_steps = args.epochs * (len(dataset) // (args.batch_size * args.accumulation_steps))

    # gradient parameters for classifier and discriminator
    classifier_gradients = []
    discriminator_gradients = []

    accumulated_error_real_d = 0
    accumulated_error_fake_d = 0
    accumulated_error_generator = 0

    accumulated_error_real_c = 0
    accumulated_error_fake_c = 0

    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))
    logger.info('There will be {} optimization steps.'.format(args.optimization_steps))

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))

        # training tracking
        num_discriminator_seen, num_classification_seen = 0, 0
        num_training_correct_real_classifier = 0
        num_training_correct_real_discriminator, num_training_correct_fake_discriminator = 0, 0
        num_training_correct_generator_classifier, num_training_correct_generator_discriminator = 0, 0

        # baseline dict: (prev baseline, total reward, num_steps)
        baseline_dict = {'prev': 0, 'total_mean_rewards': 0, 'num_steps': 0}

        for batch_iterate, batch in enumerate(epoch_iterator):
            # logger.info('Epoch: {} Iterate: {}'.format(epoch, batch_iterate))

            batch = tuple(t.to(args.device) for t in batch)
            real_inputs = {'input_ids': batch[0],
                           'attention_mask': batch[1],
                           'token_type_ids': batch[2],
                           'my_attention_mask': batch[3],
                           'classification_labels': batch[4],
                           # 'discriminator_labels': None,  # assigned later in code
                           'sentences_type': batch[5],
                          }

            # Train generator
            generatorM.train()

            # Train classifier/ discriminator
            classifierM.train()
            discriminatorM.train()

            # this changes the 'my_attention_masks' input to highlight which words should be changed
            masked_input_ids, my_attention_mask, fake_discriminator_labels = attentionM(real_inputs['input_ids'], real_inputs['my_attention_mask'])
            assert not torch.all(torch.eq(masked_input_ids, real_inputs['input_ids']))

            # flip labels to represent the wrong answers are actually right
            fake_classification_labels, real_discriminator_labels = \
                flip_labels(classification_labels=real_inputs['classification_labels'],
                            discriminator_labels=fake_discriminator_labels)

            generator_loss, fake_input_ids, (logits_gen_d, logits_gen_c) = train_generator(args,
                                                                                           masked_input_ids,
                                                                                           real_inputs['attention_mask'],
                                                                                           my_attention_mask,
                                                                                           real_inputs['token_type_ids'],
                                                                                           real_inputs['classification_labels'],
                                                                                           real_inputs['sentences_type'],
                                                                                           generatorM,
                                                                                           classifierM,
                                                                                           discriminatorM,
                                                                                           update_step,
                                                                                           baseline_dict)

            # assign fake_input_ids to fake_inputs
            fake_input_ids = fake_input_ids.to(args.device)

            if generator_loss is not None:
                generator_loss /= args.accumulation_steps
                generator_loss.backward()

            accumulated_error_generator += generator_loss.detach().item()

            num_training_correct_generator_discriminator += int(torch.sum(
                torch.eq(my_attention_mask, tanh(logits_gen_d).sign())[
                    my_attention_mask.nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())

            # print('generator model')
            # generator_grads = [p.grad for p in generatorM.parameters()]
            # for i in range(len(generator_grads)):
            #     print(i)
            #     print(generator_grads[i])
            #     if generator_grads[i] is not None:
            #         print(torch.max(generator_grads[i]))

            # zero out classifier and discriminator so they do not accumulate these gradients
            classifierM.zero_grad()
            discriminatorM.zero_grad()

            classifierM.train()
            discriminatorM.train()

            # detach the inputs so the gradient graphs don't reach back, only need them for discriminator/ classifier
            fake_input_ids, my_attention_mask = detach_inputs([fake_input_ids, my_attention_mask])

            logits, errors = train_classifier_discriminator(args, fake_input_ids, real_inputs['input_ids'], real_inputs['attention_mask'],
                                               my_attention_mask, real_inputs['token_type_ids'], real_inputs['classification_labels'],
                                               real_inputs['sentences_type'], discriminatorM, classifierM)

            # error_real_c is classification error, error_real_d is discriminator error
            if errors['error_real_c'] is not None:
                m_ = (args.classification_start - args.classification_stop) / (
                            args.optimization_steps * (2 * args.classification_squeeze - 1))
                b_ = (args.classification_squeeze * (args.classification_stop + args.classification_start) - args.classification_start) / (
                            2 * args.classification_squeeze - 1)
                classifcation_decay = max(args.classification_start, min(args.classification_stop, m_ * update_step + b_))

                errors['error_real_c'] *= (1 - classifcation_decay)
                errors['error_real_c'] /= args.accumulation_steps * args.classifier_hop_steps  # already dividing by sum in utils_classifier
                errors['error_real_c'].backward()

                accumulated_error_real_c += errors['error_real_c'].detach().item()

                errors['error_fake_c'] *= classifcation_decay
                errors['error_fake_c'] /= args.accumulation_steps * args.classifier_hop_steps
                errors['error_fake_c'].backward()

                accumulated_error_fake_c += errors['error_fake_c'].detach().item()

                num_classification_seen += sum(real_inputs['sentences_type'])
                num_training_correct_real_classifier += torch.sum(
                    torch.eq(torch.argmax(real_inputs['classification_labels'][real_inputs['sentences_type'].nonzero().squeeze()],
                                          dim=-1),
                             torch.argmax(logits['logits_real_c'][real_inputs['sentences_type'].nonzero().squeeze()], dim=-1)),
                    dtype=torch.int).detach().cpu()
                num_training_correct_generator_classifier += torch.sum(torch.eq(
                    torch.argmax(
                        real_inputs['classification_labels'][real_inputs['sentences_type'].nonzero().squeeze()],
                        dim=-1),
                    torch.argmax(logits_gen_c[real_inputs['sentences_type'].nonzero().squeeze()], dim=-1)),
                    dtype=torch.int).detach().cpu()

            num_discriminator_seen += int(my_attention_mask.nonzero().shape[0])

            if errors['error_real_d'] is not None:
                errors['error_real_d'] /= args.accumulation_steps * args.discriminator_hop_steps
                errors['error_real_d'].backward()

                accumulated_error_real_d += errors['error_real_d'].detach().item()

                num_training_correct_real_discriminator += int(torch.sum(
                    torch.eq(my_attention_mask, tanh(logits['logits_real_d']).sign())[
                        my_attention_mask.nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())

            if errors['error_fake_d'] is not None:
                errors['error_fake_d'] /= args.accumulation_steps * args.discriminator_hop_steps
                errors['error_fake_d'].backward()

                accumulated_error_fake_d += errors['error_fake_d'].detach().item()

                num_training_correct_fake_discriminator += int(torch.sum(
                    torch.eq(-1*my_attention_mask, tanh(logits['logits_fake_d']).sign())[
                        my_attention_mask.nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())

            # save gradient parameters in a list
            # for classifier
            if args.train_classifier and errors['error_real_c'] is not None:
                if not len(classifier_gradients):
                    for p in classifierM.parameters():
                        if p.grad is not None:
                            classifier_gradients.append(p.grad.clone())
                else:
                    for i, p in enumerate(classifierM.parameters()):
                        if p.grad is not None:
                            assert p.grad.shape == classifier_gradients[i].shape
                            classifier_gradients[i] += p.grad.clone()
            # for discriminator
            if args.train_discriminator:
                if not len(discriminator_gradients):
                    for p in discriminatorM.parameters():
                        if p.grad is not None:
                            discriminator_gradients.append(p.grad.clone())
                else:
                    for i, p in enumerate(discriminatorM.parameters()):
                        if p.grad is not None:
                            assert p.grad.shape == discriminator_gradients[i].shape
                            discriminator_gradients[i] += p.grad.clone()

            # zero out classifier and discriminator so they do not accumulate these gradients
            # will instead save them in a list
            classifierM.zero_grad()
            discriminatorM.zero_grad()

            if (global_step + 1) % args.accumulation_steps == 0:
                messages = {}
                messages[
                    'message_generator_batch_error'] = 'The batch Generator "loss": {:.3f}'.format(
                    accumulated_error_generator)

                messages['message_classifier_batch_error'] = 'The batch Classifier errors: real {:.3f} + fake {:.3f} = {:.3f}'.format(
                    accumulated_error_real_c,
                    accumulated_error_fake_c,
                    accumulated_error_real_c +
                    accumulated_error_fake_c)

                messages[
                    'message_discriminator_batch_error'] = 'The batch Discriminator errors: real {:.3f} + fake {:.3f} = {:.3f}'.format(
                    accumulated_error_real_d,
                    accumulated_error_fake_d,
                    accumulated_error_fake_d +
                    accumulated_error_real_d)

                if num_classification_seen:
                    messages['message_classifier_real_running_total_correct'] = '\treal Classification is {} out of {} for a percentage of {:.3f}'.format(
                            num_training_correct_real_classifier, num_classification_seen,
                            num_training_correct_real_classifier / float(num_classification_seen))

                    messages['message_classifier_generator_running_total_correct'] = '\tGenerator Classification is {} out of {} for a percentage of {:.3f}'.format(
                            num_training_correct_generator_classifier, num_classification_seen,
                            num_training_correct_generator_classifier / float(num_classification_seen))
                else:
                    messages['message_classifier'] = '\tNo multiple choice labels seen yet.'

                messages[
                    'message_generator_discriminator_running_total_correct'] = '\tGenerator Discriminator is {} out of {} for a percentage of {:.3f}'.format(
                    num_training_correct_generator_discriminator, num_discriminator_seen,
                    num_training_correct_generator_discriminator / float(num_discriminator_seen)
                )
                messages[
                    'message_discriminator_real_running_total_correct'] = '\treal Discriminator is {} out of {} for a percentage of {:.3f}'.format(
                    num_training_correct_real_discriminator, num_discriminator_seen,
                    num_training_correct_real_discriminator / float(num_discriminator_seen)
                )
                messages[
                    'message_dicriminator_fake_running_total_correct'] = '\tfake Discriminator is {} out of {} for a percentage of {:.3f}'.format(
                    num_training_correct_fake_discriminator, num_discriminator_seen,
                    num_training_correct_fake_discriminator / float(num_discriminator_seen)
                )

                if not args.quiet:
                    logger.info('Running totals for current epoch {}, after update {}'.format(epoch,
                                                                                              update_step))
                    # print('\nRunning totals for current epoch {}, after update {}'.format(epoch,
                    #                                                                     update_step))
                    for m in messages.values():
                        logger.info(m)
                        # print(m)

                if args.do_ablation_discriminator:
                    ablation_dir = os.path.join(args.output_dir, 'ablation_train')

                    if not os.path.exists(ablation_dir):
                        os.makedirs(ablation_dir)

                    ablation_filename = os.path.join(ablation_dir, 'checkpoint_{}_{}.txt'.format(update_step, '-'.join(
                        args.domain_words)))
                    if os.path.exists(ablation_filename):
                        os.remove(ablation_filename)

                    ablation_fake_inputs = {'input_ids': fake_input_ids,
                                            'my_attention_mask': my_attention_mask}
                    ablation_real_inputs = {'input_ids': real_inputs['input_ids'],
                                            'attention_mask': real_inputs['attention_mask']}

                    ablation_discriminator(args, ablation_filename, tokenizer, ablation_fake_inputs, ablation_real_inputs,
                                           tanh(logits['logits_fake_d']), tanh(logits['logits_real_d']), tanh(logits_gen_d))

                accumulated_error_real_d = 0
                accumulated_error_fake_d = 0
                accumulated_error_generator = 0

                accumulated_error_real_c = 0
                accumulated_error_fake_c = 0

                if args.train_generator:
                    # Update generatorM parameters
                    # print('generator model')
                    # for i in range(len(list(generatorM.parameters()))):
                    #     print(i)
                    #     print(list(generatorM.parameters())[i].grad)

                    generatorO.step()

                    if args.do_generator_weight_clipping:
                        generatorM._modules['correct_elementwise'].apply(weight_clipper)
                        generatorM._modules['incorrect_elementwise'].apply(weight_clipper)

                # update classifier/discriminator parameters
                if (update_step + 1) % args.classifier_hop_steps == 0:
                    if args.train_classifier and len(classifier_gradients):
                        for i, p in enumerate(classifierM.parameters()):
                            if p.grad is not None:
                                assert p.grad.shape == classifier_gradients[i].shape
                                p.grad += classifier_gradients[i]
                        classifierO.step()
                        classifier_gradients = []
                if (update_step + 1) % args.discriminator_hop_steps == 0:
                    if args.train_discriminator:
                        for i, p in enumerate(discriminatorM.parameters()):
                            if p.grad is not None:
                                assert p.grad.shape == discriminator_gradients[i].shape
                                p.grad += discriminator_gradients[i]
                        discriminatorO.step()
                        discriminator_gradients = []

                # zero out networks
                classifierM.zero_grad()
                discriminatorM.zero_grad()
                generatorM.zero_grad()

                update_step += 1

                if args.save_steps > 0 and (update_step + 1) % args.save_steps == 0:

                    assert save_models(args, update_step+1, generatorM, classifierM, discriminatorM) == -1

                    if args.evaluate_during_training:
                        assert evaluate(args, classifierM, generatorM, attentionM, tokenizer, update_step+1)

            global_step += 1

        epoch_iterator.close()
    train_iterator.close()

    return attentionM, generatorM, classifierM, update_step


def evaluate(args, classifierM, generatorM, attentionM, tokenizer, checkpoint, test=False):
    results = {}
    subset = 'test' if test else 'dev'

    # change models to eval
    classifierM.eval()
    generatorM.eval()
    attentionM.eval()

    eval_dataset = load_and_cache_features(args, tokenizer, subset)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    if args.do_ablation_classifier:
        ablation_dir = os.path.join(args.output_dir, 'ablation_{}'.format(subset))

        if not os.path.exists(ablation_dir):
            os.makedirs(ablation_dir)

        ablation_filename = os.path.join(ablation_dir, 'checkpoint_{}_{}.txt'.format(checkpoint, '-'.join(args.domain_words)))

        if os.path.exists(ablation_filename):
            os.remove(ablation_filename)

    logger.info('Starting Evaluation!')
    logger.info('Number of examples: {}'.format(len(eval_dataset)))
    logger.info('Relevant Checkpoint is {}'.format(checkpoint))
    # print('Starting Evaluation!\nNumber of examples: {}.\nRelevant Checkpoint is {}.'.format(len(eval_dataset),
    #                                                                                          checkpoint))

    real_loss = 0.
    fake_loss = 0.
    all_real_predictions = None
    all_fake_predictions = None
    all_labels = None
    num_steps = 0
    all_saved_data = {}

    num_batches = len(eval_dataloader)

    with torch.no_grad():
        for batch_ind, batch in tqdm(enumerate(eval_dataloader),
                                     'Evaluating {} batches of batch size {} from subset {}'.format(num_batches,
                                                                                                    args.batch_size,
                                                                                                    subset)):
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'my_attention_mask': batch[3],
                      'classification_labels': batch[4],
                      # 'discriminator_labels': None, # should be assigned within
                      'sentences_type': batch[5],
                      }

            masked_input_ids, my_attention_mask, fake_discriminator_labels = attentionM(inputs['input_ids'], inputs['my_attention_mask'])
            # fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}
            fake_vocabulary_probs = generatorM(masked_input_ids, inputs['attention_mask'],
                                               inputs['token_type_ids'], inputs['classification_labels'])

            # choose words at random or argmax or based on probability
            fake_action, _ = word_index_selector(fake_vocabulary_probs, 'sample_log', -1)
            fake_input_ids = (my_attention_mask * fake_action +
                              (1 - my_attention_mask) * inputs['input_ids']).long()

            # flip labels to represent the wrong answers are actually right
            # fake_classification_labels, _ = flip_labels(inputs['classification_labels'], None)

            fake_predictions = classifierM(fake_input_ids, inputs['attention_mask'], inputs['token_type_ids'])
            real_predictions = classifierM(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

            fake_error = my_CrossEntropyLoss(fake_predictions, inputs['classification_labels'].nonzero()[:, 1].long(),
                                             neg_one=False)
            real_error = my_CrossEntropyLoss(real_predictions, inputs['classification_labels'].nonzero()[:, 1].long(),
                                             neg_one=True)

            real_loss += torch.sum(real_error).item()
            fake_loss += torch.sum(fake_error).item()

            if all_real_predictions is None:
                all_real_predictions = real_predictions.detach().cpu().numpy()
                all_fake_predictions = fake_predictions.detach().cpu().numpy()
                all_labels = inputs['classification_labels'].detach().cpu().numpy()
            else:
                all_real_predictions = np.append(all_real_predictions, real_predictions.detach().cpu().numpy(), axis=0)
                all_fake_predictions = np.append(all_fake_predictions, fake_predictions.detach().cpu().numpy(), axis=0)
                all_labels = np.append(all_labels, inputs['classification_labels'].detach().cpu().numpy(), axis=0)

            if args.do_ablation_classifier:  # and batch_ind in ablation_indices:
                ablation_fake_inputs = {'input_ids': fake_input_ids,
                                        'my_attention_mask': my_attention_mask,
                                        'classification_labels': inputs['classification_labels']}
                assert -1 == ablation(args, ablation_filename, tokenizer, ablation_fake_inputs, inputs,
                                      real_predictions, fake_predictions)

            num_steps += 1

    real_eval_loss = real_loss / num_steps
    fake_eval_loss = fake_loss / num_steps
    cum_real_predictions = np.argmax(all_real_predictions, axis=1)
    cum_fake_predictions = np.argmax(all_fake_predictions, axis=1)
    all_labels = np.argmax(all_labels, axis=1)
    real_accuracy = accuracy_score(all_labels, cum_real_predictions)
    real_num_correct = accuracy_score(all_labels, cum_real_predictions, normalize=False)
    fake_accuracy = accuracy_score(all_labels, cum_fake_predictions)
    fake_num_correct = accuracy_score(all_labels, cum_fake_predictions, normalize=False)

    results['real accuracy'] = real_accuracy
    results['number correct real classifications'] = real_num_correct
    results['fake accuracy'] = fake_accuracy
    results['number correct fake classifications'] = fake_num_correct
    results['number of examples'] = len(eval_dataset)
    results['real loss'] = round(real_eval_loss, 4)
    results['fake loss'] = round(fake_eval_loss, 4)

    logger.info('After evaluating:')
    for key, val in results.items():
        logger.info('The {} is {}'.format(key, round(val, 3)))
        # print('The {} is {}'.format(key, round(val, 3)))

    with open(args.results_filename, 'r') as rf:
        opener = next(rf)
        all_score_tuples = [make_tuple(line) for ind, line in enumerate(rf) if ind > 0 and bool(line)]
        rf.close()

    all_score_tuples.append((real_accuracy, args.logging_file_number, checkpoint))
    all_score_tuples.sort(key=lambda t: t[0], reverse=True)

    with open(args.results_filename, 'w') as rf:
        rf.write('{}\n'.format(opener))
        for tup in all_score_tuples:
            rf.write('{}\n'.format(str(tup)))
        rf.close()

    return -1


def main():
    class Args(object):
        def __init__(self):
            self.data_dir = '../ARC/ARC-with-context/'
            self.output_dir = 'output/'
            self.cache_dir = 'saved/'

            self.transformer_name = 'bert'
            self.tokenizer_name = 'bert-base-uncased'
            self.generator_model_type = 'bert-reinforce'
            self.generator_model_name = '/home/kinne174/private/Output/transformers_gpu/language_modeling/saved/all_bert-base-uncased_v2/'
            # self.generator_model_name = '/home/kinne174/private/PythonProjects/gan_qa/output/bert-reinforce-bert-reinforce-/bert-generator-105/'
            self.attention_model_type = 'essential-reinforce'

            self.classifier_model_type = 'bert-reinforce'
            self.classifier_model_name = '/home/kinne174/private/PythonProjects/gan_qa/output/bert-reinforce-bert-reinforce-/saved/my_method/21/bert-classifier-70/'
            # self.classifier_model_name = '/home/kinne174/private/Output/transformers_gpu/classification/saved/bert-base-uncased-r/'
            # self.classifier_model_name = None
            self.classifier_embedding_dim = 50
            self.classifier_hidden_dim = 512
            self.classifier_num_layers = 7

            self.discriminator_model_type = 'lstm-reinforce'
            # self.discriminator_model_name = '/home/kinne174/private/PythonProjects/gan_qa/output/roberta-reinforce-roberta-reinforce-/saved/my_method/roberta-discriminator-140/'
            self.discriminator_model_name = None
            self.discriminator_embedding_type = None
            self.discriminator_embedding_dim = 50
            self.discriminator_hidden_dim = 512
            self.discriminator_num_layers = 2
            self.discriminator_dropout = 0.10

            self.epochs = 12
            self.cutoff = None
            self.do_evaluate_test = False
            self.do_evaluate_dev = True
            self.do_train = True
            self.use_gpu = True
            self.overwrite_output_dir = True
            self.overwrite_cache_dir = False
            self.seed = 1234
            self.max_length = 256
            self.do_lower_case = True
            self.essential_terms_hidden_dim = 512
            self.evaluate_all_models = False
            self.quiet = False
            self.cuda_num = 7

            self.domain_words = []
            self.evaluate_during_training = True
            self.learning_rate_classifier = 9e-5
            self.learning_rate_generator = 9e-6
            self.learning_rate_discriminator = 9.5e-4
            self.batch_size = 5
            self.save_steps = 15
            self.essential_mu_p = 0.25
            self.use_corpus = False
            self.accumulation_steps = 100
            self.classifier_hop_steps = 10
            self.discriminator_hop_steps = 3
            self.train_classifier = False
            self.train_discriminator = True
            self.train_generator = True
            self.reinforce_gamma = 0.8
            self.do_ablation_discriminator = True
            self.do_ablation_classifier = True
            self.do_prevaluation = False
            self.lambda_baseline = 0.15
            self.rewards_stop = 0.85
            self.rewards_start = 0.8
            self.rewards_squeeze = 0.1
            self.classification_stop = 0.00
            self.classification_start = 0.00
            self.classification_squeeze = 0.20
            self.generator_resampling_steps = 5
            self.do_generator_weight_clipping = True

            self.oracleM_directory = None

        # TODO pre train discriminator on generator while not updating generator parameters,
        #  look into getting probabilities of generator from only masking current token and leaving everything else
        #  look into every so often drawing completely at random

    args = Args()

    # Setup logging
    if not os.path.isdir('logging/{}'.format('_'.join(args.domain_words))):
        os.makedirs('logging/{}'.format('_'.join(args.domain_words)))
    num_logging_files = len(glob.glob('logging/{}/logging_g-{}_c-{}*'.format('_'.join(args.domain_words) if args.domain_words else 'all',
                                                                             args.generator_model_type,
                                                                             args.classifier_model_type)))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler('logging/{}/logging_g-{}_c-{}-{}'.format(
                                                      '_'.join(args.domain_words) if args.domain_words else 'all',
                                                      args.generator_model_type,
                                                      args.classifier_model_type,
                                                      num_logging_files)),
                                  logging.StreamHandler()])
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO,
    #                     filename='logging/{}/logging_g-{}_c-{}-{}'.format('_'.join(args.domain_words) if args.domain_words else 'all',
    #                                                                       args.generator_model_type,
    #                                                                       args.classifier_model_type,
    #                                                                       num_logging_files))
    args.logging_file_number = num_logging_files

    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))

    folder_name_output = '-'.join([args.generator_model_type, args.classifier_model_type, '_'.join(args.domain_words)])
    proposed_output_dir = os.path.join(args.output_dir, folder_name_output, str(args.logging_file_number))
    if not os.path.exists(proposed_output_dir):
        os.makedirs(proposed_output_dir)
    else:
        if os.listdir(proposed_output_dir):
            if not args.overwrite_output_dir:
                raise Exception(
                    "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                        proposed_output_dir))

    # set up results folder before changing output_dir
    results_filename = os.path.join(args.output_dir, folder_name_output, 'results/results.txt')
    if not os.path.exists(results_filename):
        os.makedirs(os.path.join(args.output_dir,  folder_name_output, 'results'))
        with open(results_filename, 'w') as rf:
            rf.write('Results for {}'.format('-'.join(args.domain_words)))
            rf.close()

    args.results_filename = results_filename

    args.output_dir = proposed_output_dir

    # reassign cache dir based on domain words
    folder_name_cache = '-'.join([args.transformer_name])
    proposed_cache_dir = os.path.join(args.cache_dir, folder_name_cache)
    if not os.path.exists(proposed_cache_dir):
        os.makedirs(proposed_cache_dir)

    args.cache_dir = proposed_cache_dir

    # print out arguments
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    # get tokenizer
    tokenizer_class = TOKENIZER_CLASSES[args.transformer_name]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # get whether running on cpu or gpu
    # device = get_device() if args.use_gpu else torch.device('cpu')
    device = torch.device('cuda:{}'.format(args.cuda_num))
    args.device = device
    logger.info('Using device {}'.format(args.device))

    models_checkpoints = []
    if args.do_train:
        # dataset includes: input_ids [n, 4, max_length], input_masks [n, 4, max_length], token_type_masks [n, 4, max_length]
        # attention_masks [n, 4, max_length] and labels [n, 4]
        dataset = load_and_cache_features(args, tokenizer, 'train')

        # initialize and return models
        generatorM, attentionM, classifierM, discriminatorM = inititalize_models(args, tokenizer)

        attentionM, generatorM, classifierM, update_step = train(args, tokenizer, dataset, generatorM, attentionM, classifierM, discriminatorM)
        models_checkpoints = [((attentionM, generatorM, classifierM), update_step)]

    if args.do_evaluate_dev:
        if not len(models_checkpoints):
            if args.evaluate_all_models:
                models_checkpoints = load_models(args, tokenizer)
            else:
                generatorM, attentionM, classifierM, _ = inititalize_models(args, tokenizer)
                models_checkpoints = [((generatorM, attentionM, classifierM), -1)]

        for (attentionM, generatorM, classifierM), cp in models_checkpoints:

            # move to proper device based on if gpu is available
            logger.info('Loading evaluation models from checkpoint {} to {}'.format(cp, args.device))
            generatorM.to(args.device)
            attentionM.to(args.device)
            classifierM.to(args.device)

            assert -1 == evaluate(args, classifierM, generatorM, attentionM, tokenizer, cp)

            assert args.oracleM_directory is not None
            saved_data = stream_data(args, generatorM, attentionM, subset='dev', oracleM_dir=args.oracleM_directory)
            discriminator_filename = os.path.join(args.output_dir, 'generator_eval', 'generator_eval-{}.txt'.format(cp))
            assert -1 == write_out_discriminator_eval(saved_data, tokenizer, discriminator_filename)


    if args.do_evaluate_test:
        models_checkpoints = load_models(args, tokenizer) if not len(models_checkpoints) else models_checkpoints

        for (attentionM, generatorM, classifierM), cp in models_checkpoints:

            # move to proper device based on if gpu is available
            logger.info('Loading evaluation models from checkpoint {} to {}'.format(cp, args.device))
            generatorM.to(args.device)
            attentionM.to(args.device)
            classifierM.to(args.device)

            assert -1 == evaluate(args, classifierM, generatorM, attentionM, tokenizer, cp, test=True)


if __name__ == '__main__':
    main()











