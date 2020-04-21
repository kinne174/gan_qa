from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import getpass
import logging
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import random

from transformers import (BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer)

from utils_real_data import example_loader
from utils_embedding_model import feature_loader, load_features, save_features
from utils_classifier import flip_labels
from utils_generator import generator_models_and_config_classes
from utils_model_maitenence import inititalize_models, save_models, load_models
from utils_ablation import ablation, ablation_discriminator

# logging
logger = logging.getLogger(__name__)

# sigmoid
sigmoid = torch.nn.Sigmoid()

# hugging face transformers default models, can use pretrained ones though too
TOKENIZER_CLASSES = {
    'roberta': RobertaTokenizer,
    'albert': AlbertTokenizer,
}


def set_seed(args):
    np.random.seed(args.seed)
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


def detach_inputs(fake_inputs, inputs):
    assert 'input_ids' in fake_inputs and 'input_ids' in inputs
    # assert 'inputs_embeds' in fake_inputs

    fake_inputs = {k: v.detach() if hasattr(v, 'detach') else v for k, v in fake_inputs.items()}
    inputs = {k: v.detach() if hasattr(v, 'detach') else v for k, v in inputs.items()}

    return fake_inputs, inputs


# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:5')
    else:
        return torch.device('cpu')


# returns a list of length num_choices with each entry a length four list with a 1 in the correct response spot and 0s elsewhere
def label_map(labels, num_choices):

    def label_list(target, new_target, switch_index, num_choices):
        l = [target]*num_choices
        l[switch_index] = new_target
        return l

    answers = [label_list(0, 1, lab, num_choices) for lab in labels]
    return answers


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
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size*args.minibatch_size)

    # optimizers
    classifierO = AdamW(classifierM.parameters(), lr=args.learning_rate_classifier, eps=args.epsilon_classifier, weight_decay=1)
    generatorO = AdamW(generatorM.parameters(), lr=args.learning_rate_generator, eps=args.epsilon_generator, weight_decay=1)
    discriminatorO = AdamW(discriminatorM.parameters(), lr=args.learning_rate_classifier, eps=args.epsilon_classifier)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    if args.do_ablation_discriminator:
        ablation_dir = os.path.join(args.output_dir, 'ablation_train')

        if not os.path.exists(ablation_dir):
            os.makedirs(ablation_dir)

    best_dev_acc = 0.0
    global_step, update_step = 0, 0

    # do an evaluation first
    if args.do_prevaluation:
        evaluate(args, classifierM, generatorM, attentionM, tokenizer, global_step)

    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))
    logger.info('There will be {} iterations.'.format(len(dataset)//(args.batch_size*args.minibatch_size) + 1))

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size*args.minibatch_size))

        num_discriminator_seen, num_classification_seen = 0, 0
        num_training_correct_real_classifier = 0
        num_training_correct_real_discriminator, num_training_correct_fake_discriminator = 0, 0
        num_training_correct_generator_classifier, num_training_correct_generator_discriminator = 0, 0

        for batch_iterate, batch in enumerate(epoch_iterator):
            logger.info('Epoch: {} Iterate: {}'.format(epoch, batch_iterate))

            batch_dataset = TensorDataset(*batch)

            # use pytorch data loaders to cycle through the data
            minibatch_sampler = SequentialSampler(batch_dataset)
            minibatch_dataloader = DataLoader(batch_dataset, sampler=minibatch_sampler, batch_size=args.batch_size)

            minibatch_iterator = tqdm(minibatch_dataloader, desc="Minibatch, minibatch size {}".format(args.batch_size))

            minibatch_error_real_d = 0
            minibatch_error_fake_d = 0
            minibatch_error_generator_d = 0

            minibatch_error_real_c = 0
            minibatch_error_generator_c = 0

            if args.do_ablation_discriminator:

                ablation_filename = os.path.join(ablation_dir, 'checkpoint_{}_{}.txt'.format(global_step, '-'.join(args.domain_words)))

                if os.path.exists(ablation_filename):
                    os.remove(ablation_filename)

            # zero out gradient of networks
            generatorM.zero_grad()
            classifierM.zero_grad()
            discriminatorM.zero_grad()

            # gradient parameters for classifier and discriminator
            classifier_gradients = []
            discriminator_gradients = []

            for minibatch_iterate, minibatch in enumerate(minibatch_iterator):

                # TODOfixed should I be splitting it up so that the Generator and Classifier get different input?
                # I don't think so because the classifier is making a judgement on the data but not training in first pass, so should still give it a chance to see it in the second
                # also don't have enough data values to afford not to pass them
                minibatch = tuple(t.to(args.device) for t in minibatch)
                inputs = {'input_ids': minibatch[0],
                          'attention_mask': minibatch[1],
                          'token_type_ids': minibatch[2],
                          'my_attention_mask': minibatch[3],
                          'classification_labels': minibatch[4],
                          # 'discriminator_labels': None,  # assigned later in code
                          'sentences_type': minibatch[5],
                          'update_step': update_step,
                          }
                mbatch_size = inputs['input_ids'].shape[0]

                # Train generator
                generatorM.train()
                attentionM.eval()

                # Train classifier/ discriminator
                classifierM.train()
                discriminatorM.train()


            # this changes the 'my_attention_masks' input to highlight which words should be changed
                fake_inputs = attentionM(**inputs)
                fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}

                if not args.no_generator_error:

                    # this changes the 'input_ids' based on the 'my_attention_mask' input to generate words to fool classifier, already dividing by sum in utils_classifier
                    fake_inputs = generatorM(**fake_inputs)

                    # flip labels to represent the wrong answers are actually right
                    fake_inputs = flip_labels(**fake_inputs)

                    # get the predictions of which answers are the correct pairing from the classifier
                    fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}

                    predictions_g_c, errorG_c = classifierM(**fake_inputs)
                    predictions_g_d, errorG_d = discriminatorM(**fake_inputs)

                    if all((errorG_c, errorG_d)) is None:
                        logger.warning('ErrorG and ErrorC is None!')
                        raise Exception('ErrorG and ErrorC is None!')

                    # based on the loss function update the parameters within the generator/ attention model

                    #errorG_c is classification error, errorG_d is discriminator error
                    if errorG_c is not None:
                        num_training_correct_generator_classifier += torch.sum(torch.eq(torch.argmin(fake_inputs['classification_labels'][fake_inputs['sentences_type'].nonzero().squeeze()], dim=-1),
                                                                               torch.argmin(predictions_g_c[fake_inputs['sentences_type'].nonzero().squeeze()], dim=-1)),
                                                                               dtype=torch.int).detach().cpu()

                        errorG_c /= args.minibatch_size
                        errorG_c.backward(retain_graph=True)

                        minibatch_error_generator_c += errorG_c.detach().item()

                    num_training_correct_generator_discriminator += int(torch.sum(torch.eq(fake_inputs['discriminator_labels'], predictions_g_d.sign())[fake_inputs['my_attention_mask'].nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())

                    errorG_d /= args.minibatch_size
                    errorG_d.backward()

                    minibatch_error_generator_d += errorG_d.detach().item()

                    # zero out classifier and discriminator so they do not accumulate these gradients
                    classifierM.zero_grad()
                    discriminatorM.zero_grad()

                # print('generator model')
                # for i in range(len(list(generatorM.parameters()))):
                #     print(i)
                #     print(list(generatorM.parameters())[i].grad)
                #     print(torch.max(list(generatorM.parameters())[i].grad))
                # print('classifier model')
                # for i in range(len(list(classifierM.parameters()))):
                #     print(i)
                #     print(list(classifierM.parameters())[i].grad)
                # print(torch.max(list(classifierM.parameters())[0].grad))
                # print('*****************************************************************')

                # detach the inputs so the gradient graphs don't reach back, only need them for classifier
                fake_inputs, inputs = detach_inputs(fake_inputs, inputs)

                # give inputs the my_attention_mask to be used in discriminator
                inputs['my_attention_mask'] = fake_inputs['my_attention_mask']
                inputs['discriminator_labels'] = fake_inputs['discriminator_labels']  # before flipping

                # flip labels to represent the wrong answers are actually right
                fake_inputs = flip_labels(**fake_inputs)
                fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}

                # see if the classifier can determine difference between fake and real data
                predictions_real_c, error_real_c = classifierM(**inputs)
                predictions_real_d, error_real_d = discriminatorM(**inputs)

                predictions_fake_d, error_fake_d = discriminatorM(**fake_inputs)

                if all((error_real_c, error_real_d, error_fake_d)) is None:
                    logger.warning('Error for classifier and discriminator is None!')
                    raise Exception('Error for classifier and discriminator is None!')

                # error_real_c is classification error, error_real_d is discriminator error
                if error_real_c is not None:

                    num_classification_seen += sum(inputs['sentences_type'])
                    num_training_correct_real_classifier += torch.sum(
                        torch.eq(torch.argmax(inputs['classification_labels'][inputs['sentences_type'].nonzero().squeeze()], dim=-1),
                                 torch.argmax(predictions_real_c[inputs['sentences_type'].nonzero().squeeze()], dim=-1)),
                                 dtype=torch.int).detach().cpu()

                    error_real_c /= args.minibatch_size  # already dividing by sum in utils_classifier
                    error_real_c.backward()

                    minibatch_error_real_c += error_real_c.detach().item()

                num_discriminator_seen += int(inputs['discriminator_labels'].nonzero().shape[0])
                num_training_correct_real_discriminator += int(torch.sum(
                    torch.eq(inputs['discriminator_labels'], predictions_real_d.sign())[
                        inputs['my_attention_mask'].nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())
                num_training_correct_fake_discriminator += int(torch.sum(
                    torch.eq(fake_inputs['discriminator_labels'], predictions_fake_d.sign())[
                        fake_inputs['my_attention_mask'].nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())

                error_real_d /= args.minibatch_size
                error_real_d.backward()

                error_fake_d /= args.minibatch_size
                error_fake_d.backward()

                minibatch_error_real_d += error_real_d.detach().item()
                minibatch_error_fake_d += error_fake_d.detach().item()

                # save gradient parameters in a list
                # for classifier
                if not args.no_classification_error and error_real_c is not None:
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
                if not args.no_discriminator_error:
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

                update_step += 1

                if args.do_ablation_discriminator:
                    ablation_discriminator(args, ablation_filename, tokenizer, fake_inputs, inputs, predictions_fake_d, predictions_real_d, predictions_g_d)

            minibatch_iterator.close()

            # output batch errors
            messages = {}
            messages['message_generator_batch_error'] = 'The batch Generator errors: discriminator {:.3f} + classifier {:.3f} = {:.3f}'.format(minibatch_error_generator_d,
                                                                                                                                   minibatch_error_generator_c,
                                                                                                                                   minibatch_error_generator_d +
                                                                                                                                   minibatch_error_generator_c)
            messages['message_classifier_batch_error'] = 'The batch Classifier error: {:.3f}'.format(minibatch_error_real_c)
            messages['message_discriminator_batch_error'] = 'The batch Discriminator errors: real {:.3f} + fake {:.3f} = {:.3f}'.format(minibatch_error_fake_d,
                                                                                                                           minibatch_error_real_d,
                                                                                                                           minibatch_error_fake_d +
                                                                                                                           minibatch_error_real_d)

            if not args.no_generator_error:
                if all([list(generatorM.parameters())[i].grad is None for i in range(len(list(generatorM.parameters())))]):
                    raise Exception(
                        'There is no gradient parameters for the generator (all None) in epoch {} iteration {}!'.format(
                            epoch, batch_iterate))
                if any([torch.max(torch.abs(list(generatorM.parameters())[i].grad)) == 0 for i in
                        range(len(list(generatorM.parameters()))) if list(generatorM.parameters())[i].grad is not None]):
                    logger.warning(
                        'There is some zero gradient parameters for the generator in epoch {} iteration {}!'.format(
                            epoch,
                            batch_iterate))
                    # raise Exception('There is all zero gradient parameters for the generator in epoch {} iteration {}!'.format(epoch, iterate))

            # Update generatorM parameters
            if not args.no_generator_error:
                generatorO.step()

            if len(classifier_gradients):
                if all([list(classifierM.parameters())[i].grad is None for i in
                        range(len(list(classifierM.parameters())))]):
                    raise Exception(
                        'There are no gradient parameters for the classifier (all None) in epoch {} iteration {}!'.format(
                            epoch, batch_iterate))
                if any([torch.max(torch.abs(list(classifierM.parameters())[i].grad)) == 0 for i in
                        range(len(list(classifierM.parameters()))) if list(classifierM.parameters())[i].grad is not None]):
                    logger.warning(
                        'There are some zero gradient parameters for the classifier in epoch {} iteration {}!'.format(epoch,
                                                                                                                      batch_iterate))
                    # raise Exception('There is some zero gradient parameters for the classifier in epoch {} iteration {}!'.format(epoch, iterate))

            # update classifier/discriminator parameters
            if not args.no_classification_error and len(classifier_gradients):
                for i, p in enumerate(classifierM.parameters()):
                    if p.grad is not None:
                        assert p.grad.shape == classifier_gradients[i].shape
                        p.grad += classifier_gradients[i]
                classifierO.step()

            if not args.no_discriminator_error:
                for i, p in enumerate(discriminatorM.parameters()):
                    if p.grad is not None:
                        assert p.grad.shape == discriminator_gradients[i].shape
                        p.grad += discriminator_gradients[i]
                discriminatorO.step()

            # logging for fake and real classification success
            if num_classification_seen:
                messages['message_classifier_real_running_total_correct'] = '\treal Classification is {} out of {} for a percentage of {:.3f}'.format(
                        num_training_correct_real_classifier, num_classification_seen,
                        num_training_correct_real_classifier / float(num_classification_seen))

                messages['message_classifier_generator_running_total_correct'] = '\tGenerator Classification is {} out of {} for a percentage of {:.3f}'.format(
                        num_training_correct_generator_classifier, num_classification_seen,
                        num_training_correct_generator_classifier / float(num_classification_seen))
            else:
                messages['message_classifier_generator_running_total_correct'] = '\tNo classication scores so far after batch {}'.format(batch_iterate)

            messages['message_generator_discriminator_running_total_correct'] = '\tGenerator Discriminator is {} out of {} for a percentage of {:.3f}'.format(
                num_training_correct_generator_discriminator, num_discriminator_seen,
                num_training_correct_generator_discriminator / float(num_discriminator_seen)
            )
            messages['message_discriminator_real_running_total_correct'] = '\treal Discriminator is {} out of {} for a percentage of {:.3f}'.format(
                num_training_correct_real_discriminator, num_discriminator_seen,
                num_training_correct_real_discriminator / float(num_discriminator_seen)
            )
            messages['message_dicriminator_fake_running_total_correct'] = '\tfake Discriminator is {} out of {} for a percentage of {:.3f}'.format(
                num_training_correct_fake_discriminator, num_discriminator_seen,
                num_training_correct_fake_discriminator / float(num_discriminator_seen)
            )

            for m in messages.values():
                logger.info(m)
                print(m)

            # save models in cache dir
            if global_step % args.save_steps == 0 and args.save_steps is not 0 and global_step is not 0:

                assert save_models(args, global_step, generatorM, classifierM, discriminatorM) == -1

                if args.evaluate_during_training:

                    eval_results = evaluate(args, classifierM, generatorM, attentionM, tokenizer, global_step)

                    if eval_results['accuracy'] > best_dev_acc:
                        best_dev_acc = eval_results['accuracy']
                        best_dev_loss = eval_results['loss']
                        best_step = global_step

                    logger.info(
                        'The dev accuracy during training at checkpoint {} is {}, loss is {}'.format(global_step,
                                                                                   round(eval_results['accuracy'], 3),
                                                                                   round(eval_results['loss'], 3)))


            global_step += 1
        epoch_iterator.close()
    train_iterator.close()

    # TODOfixed when out of double for loop save model checkpoints and report best dev results
    assert save_models(args, global_step, generatorM, classifierM, discriminatorM) == -1

    if args.evaluate_during_training:
        logger.info(
            'The best dev accuracy after training is {}, loss is {} at epoch {}.'.format(round(best_dev_acc, 3),
                                                                            round(best_dev_loss, 3), best_step))


def evaluate(args, classifierM, generatorM, attentionM, tokenizer, checkpoint, test=False):
    results = {}
    subset = 'test' if test else 'dev'

    eval_dataset = load_and_cache_features(args, tokenizer, subset)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=min(args.batch_size, 100))

    logger.info('Starting Evaluation!')
    logger.info('Number of examples: {}'.format(len(eval_dataset)))

    real_loss = 0.
    all_predictions = None
    all_labels = None
    num_steps = 0

    attentionM.eval()
    generatorM.eval()
    classifierM.eval()

    num_batches = len(eval_dataloader)
    ablation_indices = random.sample(range(num_batches), num_batches//10)

    if args.do_ablation_classifier:
        ablation_dir = os.path.join(args.output_dir, 'ablation_{}'.format(subset))

        if not os.path.exists(ablation_dir):
            os.makedirs(ablation_dir)

        ablation_filename = os.path.join(ablation_dir, 'checkpoint_{}_{}.txt'.format(checkpoint, '-'.join(args.domain_words)))

        if os.path.exists(ablation_filename):
            os.remove(ablation_filename)

    for batch_ind, batch in tqdm(enumerate(eval_dataloader), 'Evaluating {} batches of batch size {} from subset {}'.format(num_batches, min(args.batch_size, 100), subset)):
        classifierM.eval()
        generatorM.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'my_attention_mask': batch[3],
                      'classification_labels': batch[4],
                      # 'discriminator_labels': None, # should be assigned within
                      'sentences_type': batch[5],
                      }

            fake_inputs = attentionM(**inputs)
            fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}
            fake_inputs = generatorM(**fake_inputs)

            # flip labels to represent the wrong answers are actually right
            fake_inputs = flip_labels(**fake_inputs)

            # get the predictions of which answers are the correct pairing from the classifier
            fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}
            fake_predictions, _ = classifierM(**fake_inputs)
            real_predictions, real_error = classifierM(**inputs)

            real_loss += real_error.item()

        if args.do_ablation_classifier:# and batch_ind in ablation_indices:
            # inputs['token_type_ids'] = batch[2]
            assert -1 == ablation(args, ablation_filename, tokenizer, fake_inputs, inputs, real_predictions, fake_predictions)

        if all_predictions is None:
            all_predictions = real_predictions.detach().cpu().numpy()
            all_labels = inputs['classification_labels'].detach().cpu().numpy()
        else:
            all_predictions = np.append(all_predictions, real_predictions.detach().cpu().numpy(), axis=0)
            all_labels = np.append(all_labels, inputs['classification_labels'].detach().cpu().numpy(), axis=0)

        num_steps += 1

    eval_loss = real_loss / num_steps
    all_predictions = np.argmax(all_predictions, axis=1)
    all_labels = np.argmax(all_labels, axis=1)
    accuracy = accuracy_score(all_labels, all_predictions)
    num_correct = accuracy_score(all_labels, all_predictions, normalize=False)

    results['accuracy'] = accuracy
    results['number correct classifications'] = num_correct
    results['number of examples'] = len(eval_dataset)
    results['loss'] = round(eval_loss, 5)

    logger.info('After evaluating:')
    for key, val in results.items():
        logger.info('The {} is {}'.format(key, round(val, 3)))
        print('The {} is {}'.format(key, round(val, 3)))

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--on_command_line', action='store_true',
                        help='Invoke if running on command line')
    args_init = parser.parse_known_args()[0]

    if args_init.on_command_line:

        # Required
        parser.add_argument('--transformer_name', default=None, type=str, required=True,
                            help='Name of the transformer used in tokenizing in {}'.format(', '.join(list(TOKENIZER_CLASSES.keys()))))
        parser.add_argument('--tokenizer_name', default=None, type=str, required=True,
                            help='Name of the tokenizer to use from transformers package from a pretrained hf model')
        parser.add_argument('--domain_words', default=None, nargs='+', required=True,
                            help='Domain words to search for')

        # Optional
        parser.add_argument('--data_dir', default='../ARC/ARC-with-context/', type=str,
                            help='Folder where the data is being stored')
        parser.add_argument('--output_dir', default='output/', type=str,
                            help='Folder where output should be sent')
        parser.add_argument('--cache_dir', default='saved/', type=str,
                            help='Folder where saved models will be written')
        parser.add_argument('--generator_model_type', default='seq', type=str,
                            help='Type of the generator model to use from {}'.format(', '.join(list(generator_models_and_config_classes.keys()))))
        parser.add_argument('--generator_model_name', default=None, type=str,
                            help='Name or path to generator model.')
        parser.add_argument('--classifier_model_type', default='linear', type=str,
                            help='Name of the classifier model to use')
        parser.add_argument('--classifier_model_name', default=None, type=str,
                            help='Name or path to classifier model.')
        parser.add_argument('--classifier_hidden_dim', default=100, type=int,
                            help='The dimension of the hidden dimension of LSTM cell in linear classifier')
        parser.add_argument('--classifier_embedding_dim', default=50, type=int,
                            help='The dimension of the word embedding of the linear classifier')
        parser.add_argument('--attention_model_type', default='essential', type=str,
                            help='Name of attention model to use')
        parser.add_argument('--batch_size', type=int, default=5,
                            help='Size of each batch to be used in training')
        parser.add_argument('--minibatch_size', type=int, default=10,
                            help='Number of minibatches to do before stepping')
        parser.add_argument('--max_length', type=int, default=128,
                            help='The maximum length of the sequences allowed. This will induce cutting off or padding')
        parser.add_argument('--evaluate_during_training', action='store_true',
                            help='After each epoch test model on evaluation set')
        parser.add_argument('--cutoff', type=int, default=None,
                            help='Stop example collection at this number')
        parser.add_argument('--do_lower_case', action='store_true',
                            help='Tokenizer converts everything to lower case')
        parser.add_argument('--attention_window_size', type=int, default=10,
                            help='The window which to search for bigrams in the PMI attention network')
        parser.add_argument('--max_attention_words', type=int, default=3,
                            help='Maximum number of unique words to mask in attention newtworks')
        parser.add_argument('--essential_terms_hidden_dim', type=int, default=512,
                            help='Number of hidden dimensions in essential terms model')
        parser.add_argument('--essential_mu_p', type=float, default=0.15,
                            help='The proportion of context ')
        parser.add_argument('--use_corpus', action='store_true',
                            help='Activate to include corpus examples in features')
        parser.add_argument('--do_ablation', action='store_true',
                            help='While training or evaluating do ablation study')
        parser.add_argument('--evaluate_all_models', action='store_true',
                            help='When evaluating test or dev set, use all relevant checkpoints')
        parser.add_argument('--no_classification_error', action='store_true',
                            help='only train discriminator')

        parser.add_argument('--epochs', default=3, type=int,
                            help='Number of epochs to run training')
        parser.add_argument('--learning_rate_classifier', default=1e-4, type=float,
                            help='Learning rate of the classifier to be used in Adam optimization')
        parser.add_argument('--learning_rate_generator', default=1e-4, type=float,
                            help='Learning rate of the generator to be used in Adam optimization')
        parser.add_argument('--epsilon_classifier', default=1e-8, type=float,
                            help='Epsilon of classifier for Adam optimizer')
        parser.add_argument('--epsilon_generator', default=1e-8, type=float,
                            help='Epsilon of generator for Adam optimizer')
        parser.add_argument('--min_temperature', default=.5, type=float,
                            help='Minimum temperature for annealing')
        parser.add_argument('--save_steps', default=50, type=int,
                            help='After this many steps save models')
        parser.add_argument('--do_evaluate_dev', action='store_true',
                            help='Use models on "dev" dataset')
        parser.add_argument('--do_evaluate_test', action='store_true',
                            help='Use models on "test" dataset')
        parser.add_argument('--do_train', action='store_true',
                            help='Train models on training dataset')
        parser.add_argument('--use_gpu', action='store_true',
                            help='Use a gpu')
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help='Overwrite the output directory')
        parser.add_argument('--overwrite_cache_dir', action='store_true',
                            help='Overwrite the cached models directory')
        parser.add_argument('--clear_output_dir', action='store_true',
                            help='Clear all files in output directory')
        parser.add_argument('--seed', default=1234, type=int,
                            help='Random seed for reproducibility')

        args = parser.parse_args()
    else:
        class Args(object):
            def __init__(self):
                self.data_dir = '../ARC/ARC-with-context/'
                self.output_dir = 'output/'
                self.cache_dir = 'saved/'
                self.tokenizer_name = 'roberta-base'
                self.generator_model_type = 'roberta'
                # self.generator_model_name = '/home/kinne174/private/Output/transformers_gpu/language_modeling/saved/moon_roberta-base/'
                self.generator_model_name = '/home/kinne174/private/Output/transformers_gpu/language_modeling/saved/moon_roberta-base/'
                # self.generator_model_name = 'roberta-base'
                self.classifier_model_type = 'roberta'
                self.classifier_model_name = '/home/kinne174/private/Output/transformers_gpu/classification/saved/roberta-base/'
                self.attention_model_type = 'essential'
                self.transformer_name = 'roberta'
                self.evaluate_during_training = True
                self.cutoff = None
                self.epochs = 3
                self.learning_rate_classifier = 9e-5
                self.learning_rate_generator = 9e-6
                self.epsilon_classifier = 1e-9
                self.epsilon_generator = 1e-9
                self.do_evaluate_dev = True
                self.do_evaluate_test = False
                self.do_train = True
                self.use_gpu = True
                self.overwrite_output_dir = True
                self.overwrite_cache_dir = False
                self.clear_output_dir = False
                self.seed = 1234
                self.max_length = 256
                self.batch_size = 2
                self.do_lower_case = True
                self.save_steps = 5
                self.essential_terms_hidden_dim = 512
                self.essential_mu_p = 0.25
                self.use_corpus = False
                self.evaluate_all_models = False
                self.do_ablation_classifier = True
                self.do_ablation_discriminator = False
                self.domain_words = ['moon', 'earth']
                self.minibatch_size = 25
                self.classifier_hidden_dim = 100
                self.classifier_embedding_dim = 10
                self.no_classification_error = False
                self.no_discriminator_error = False
                self.no_generator_error = False
                self.discriminator_model_type = 'lstm'
                # self.discriminator_model_name = '/home/kinne174/private/PythonProjects/gan_qa/output/roberta-roberta-moon_earth/saved/0/roberta-discriminator/'
                self.discriminator_model_name = None
                self.discriminator_embedding_type = None
                self.discriminator_embedding_dim = 50
                self.discriminator_hidden_dim = 512
                self.discriminator_num_layers = 2
                self.discriminator_dropout = 0.10
                self.do_prevaluation = True

        args = Args()

    # Setup logging
    num_logging_files = len(glob.glob('logging/logging_g-{}_c-{}_{}*'.format(args.generator_model_type, args.classifier_model_type, '_'.join(args.domain_words))))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/logging_g-{}_c-{}_{}-{}'.format(args.generator_model_type, args.classifier_model_type,
                                                                          '_'.join(args.domain_words), num_logging_files))

    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))
    if (not args.do_train) and (args.do_evaluate_test or args.do_evaluate_dev) and args.clear_output_dir:
        raise Exception('You are clearing the output directory without training and asking to evaluate on the test and/or dev set!\n'
                        'Fix one of --train, --evaluate_test, --evaluate_dev, or --clear_output_dir')

    # within output and saved folders create a folder with domain words to keep output and saved objects
    folder_name_output = '-'.join([args.generator_model_type, args.classifier_model_type, '_'.join(args.domain_words)])
    proposed_output_dir = os.path.join(args.output_dir, folder_name_output)
    if not os.path.exists(proposed_output_dir):
        os.makedirs(proposed_output_dir)
    else:
        if os.listdir(proposed_output_dir):
            if not args.overwrite_output_dir:
                raise Exception(
                    "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                        proposed_output_dir))
            elif args.clear_output_dir:
                for folder in os.listdir(proposed_output_dir):
                    filenames = os.listdir(os.path.join(proposed_output_dir, folder))
                    for filename in filenames:
                        file_path = os.path.join(proposed_output_dir, folder, filename)
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            logger.info('Failed to delete {}. Reason: {}'.format(file_path, e))
                    os.rmdir(os.path.join(proposed_output_dir, folder))
    if not args.overwrite_output_dir and args.clear_output_dir:
        logger.info('If you want to clear the output directory make sure to set --overwrite_output_dir too')

    args.output_dir = proposed_output_dir

    # reassign cache dir based on domain words
    folder_name_cache = '-'.join([args.transformer_name])
    proposed_cache_dir = os.path.join(args.cache_dir, folder_name_cache)
    if not os.path.exists(proposed_cache_dir):
        os.makedirs(proposed_cache_dir)

    args.cache_dir = proposed_cache_dir

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    # get tokenizer
    tokenizer_class = TOKENIZER_CLASSES[args.transformer_name]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # get whether running on cpu or gpu
    device = get_device() if args.use_gpu else torch.device('cpu')
    args.device = device
    logger.info('Using device {}'.format(args.device))

    if args.do_train:
        # initialize and return models
        generatorM, attentionM, classifierM, discriminatorM = inititalize_models(args, tokenizer)

        # dataset includes: input_ids [n, 4, max_length], input_masks [n, 4, max_length], token_type_masks [n, 4, max_length]
        # attention_masks [n, 4, max_length] and labels [n, 4]
        dataset = load_and_cache_features(args, tokenizer, 'train')

        train(args, tokenizer, dataset, generatorM, attentionM, classifierM, discriminatorM)


    if args.do_evaluate_dev:
        models_checkpoints = load_models(args, tokenizer)
        for (attentionM, generatorM, classifierM), cp in models_checkpoints:

            # move to proper device based on if gpu is available
            logger.info('Loading evaluation models from checkpoint {} to {}'.format(cp, args.device))
            generatorM.to(args.device)
            attentionM.to(args.device)
            classifierM.to(args.device)

            eval_results = evaluate(args, classifierM, generatorM, attentionM, tokenizer, cp)
            logger.info('The dev accuracy for checkpoint {} is {}, loss is {}'.format(cp, round(eval_results['accuracy'], 3),
                                                                                       round(eval_results['loss'], 3)))
            print('The dev accuracy for checkpoint {} is {}, loss is {}'.format(cp, round(eval_results['accuracy'], 3),
                                                                                       round(eval_results['loss'], 3)))

    if args.do_evaluate_test:
        models_checkpoints = load_models(args, tokenizer)
        for (attentionM, generatorM, classifierM), cp in models_checkpoints:
            eval_results = evaluate(args, classifierM, generatorM, attentionM, tokenizer, cp, test=True)
            logger.info('The test accuracy for checkpoint {} is {}, loss is {}'.format(cp, round(eval_results['accuracy'], 3),
                                                                                       round(eval_results['loss'], 3)))


if __name__ == '__main__':
    main()
