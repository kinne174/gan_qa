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
import random
import numpy as np
from sklearn.metrics import accuracy_score


from transformers import (RobertaTokenizer, AlbertTokenizer)

from utils_embedding_model import feature_loader, load_features, save_features
from utils_real_data import example_loader
from utils_reinforce import train_classifier_discriminator, train_generator, word_index_selector, my_CrossEntropyLoss
from utils_model_maitenence import save_models, load_models, inititalize_models




# logging
logger = logging.getLogger(__name__)

# hugging face transformers default models, can use pretrained ones though too
TOKENIZER_CLASSES = {
    'roberta': RobertaTokenizer,
    'albert': AlbertTokenizer,
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


def detach_inputs(fake_inputs, inputs):
    assert 'input_ids' in fake_inputs and 'input_ids' in inputs
    assert 'inputs_embeds' in fake_inputs

    fake_inputs = {k: v.detach() if hasattr(v, 'detach') else v for k, v in fake_inputs.items()}
    inputs = {k: v.detach() if hasattr(v, 'detach') else v for k, v in inputs.items()}

    return fake_inputs, inputs


# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:4')
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


def flip_labels(classification_labels, discriminator_labels):
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
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size*args.accumulation_steps)

    # optimizers
    classifierO = AdamW(classifierM.parameters(), lr=args.learning_rate_classifier)
    generatorO = AdamW(generatorM.parameters(), lr=args.learning_rate_generator)
    discriminatorO = AdamW(discriminatorM.parameters(), lr=args.learning_rate_classifier)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    update_step = 0

    # zero out gradient of networks
    generatorM.zero_grad()
    classifierM.zero_grad()
    discriminatorM.zero_grad()

    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))
    logger.info('There will be {} iterations.'.format(len(dataset) // (args.batch_size * args.accumulation_steps) + 1))

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))

        for batch_iterate, batch in enumerate(epoch_iterator):
            logger.info('Epoch: {} Iterate: {}'.format(epoch, batch_iterate))

            # gradient parameters for classifier and discriminator
            classifier_gradients = []
            discriminator_gradients = []

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
            fake_inputs = attentionM(**real_inputs)

            # flip labels to represent the wrong answers are actually right
            fake_inputs = flip_labels(**fake_inputs)

            fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}

            generator_reward, fake_input_ids, (logits_discriminator, logits_classifier) = train_generator(args,
                                                                                                          fake_inputs,
                                                                                                          generatorM,
                                                                                                          classifierM,
                                                                                                          discriminatorM)

            # assign fake_input_ids to fake_inputs
            fake_input_ids = fake_input_ids.to(args.device)
            fake_inputs['input_ids'] = fake_input_ids

            generator_reward /= args.accumulation_steps
            generator_reward.backward()

            # zero out classifier and discriminator so they do not accumulate these gradients
            classifierM.zero_grad()
            discriminatorM.zero_grad()

            # detach the inputs so the gradient graphs don't reach back, only need them for discriminator/ classifier
            fake_inputs, real_inputs = detach_inputs(fake_inputs, real_inputs)

            # give inputs the my_attention_mask to be used in discriminator
            real_inputs['my_attention_mask'] = fake_inputs['my_attention_mask']
            real_inputs['discriminator_labels'] = fake_inputs['discriminator_labels']  # before flipping

            # flip labels to represent the wrong answers are actually right
            fake_inputs = flip_labels(**fake_inputs)
            fake_inputs = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in fake_inputs.items()}

            (logits_real_c, logits_real_d, logits_fake_d), (error_real_c, error_real_d, error_fake_d) = \
                train_classifier_discriminator(args, fake_inputs, real_inputs, discriminatorM, classifierM)

            # error_real_c is classification error, error_real_d is discriminator error
            if error_real_c is not None:
                error_real_c /= args.accumulation_steps  # already dividing by sum in utils_classifier
                error_real_c.backward()

            if error_real_d is not None:
                error_real_d /= args.accumulation_steps
                error_real_d.backward()

            if error_fake_d is not None:
                error_fake_d /= args.accumulation_steps
                error_fake_d.backward()

            # save gradient parameters in a list
            # for classifier
            if args.train_classifier and error_real_c is not None:
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

            if (batch_iterate + 1) % args.accumulation_steps == 0:
                # Update generatorM parameters
                generatorO.step()

                # update classifier/discriminator parameters
                if args.train_classifier and len(classifier_gradients):
                    print('classifier model')
                    for i, p in enumerate(classifierM.parameters()):
                        if p.grad is not None:
                            assert p.grad.shape == classifier_gradients[i].shape
                            p.grad += classifier_gradients[i]
                    print('classifier model')
                    classifierO.step()

                if args.train_discriminator:
                    for i, p in enumerate(discriminatorM.parameters()):
                        if p.grad is not None:
                            assert p.grad.shape == discriminator_gradients[i].shape
                            p.grad += discriminator_gradients[i]
                    discriminatorO.step()

                # zero out networks
                classifierM.zero_grad()
                discriminatorM.zero_grad()
                generatorM.zero_grad()

                update_step += 1

                if args.save_steps > 0 and (update_step + 1) % args.save_steps == 0:

                    assert save_models(args, update_step, generatorM, classifierM, discriminatorM) == -1

                    if args.evaluate_during_training:
                        pass

        epoch_iterator.close()
    train_iterator.close()

def evaluate(args, classifierM, generatorM, attentionM, tokenizer, checkpoint, test=False):
    results = {}
    subset = 'test' if test else 'dev'

    eval_dataset = load_and_cache_features(args, tokenizer, subset)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=min(args.batch_size, 100))

    logger.info('Starting Evaluation!')
    logger.info('Number of examples: {}'.format(len(eval_dataset)))

    real_loss = 0.
    fake_loss = 0.
    all_real_predictions = None
    all_fake_predictions = None
    all_labels = None
    num_steps = 0

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
            fake_vocabulary_probs = generatorM(masked_input_ids, inputs['attention_mask'], inputs['token_type_ids'])

            # choose words at random or argmax or based on probability
            fake_action = word_index_selector(fake_vocabulary_probs, 'argmax')
            fake_input_ids = (my_attention_mask * fake_action +
                              (1 - my_attention_mask) * inputs['input_ids']).long()

            # flip labels to represent the wrong answers are actually right
            fake_classification_labels, _ = flip_labels(inputs['classification_labels'], None)

            fake_predictions = classifierM(fake_input_ids, inputs['attention_mask'], inputs['token_type_ids'])
            real_predictions = classifierM(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

            fake_error = my_CrossEntropyLoss(fake_predictions, neg_one=False)[(fake_classification_labels == 0).nonzero(as_tuple=True)]
            real_error = my_CrossEntropyLoss(real_predictions, neg_one=True)[inputs['classification_labels'].nonzero(as_tuple=True)]

            real_loss += real_error.item()
            fake_loss += fake_error.item()

        if all_real_predictions is None:
            all_real_predictions = real_predictions.detach().cpu().numpy()
            all_fake_predictions = fake_predictions.detach().cpu().numpy()
            all_labels = inputs['classification_labels'].detach().cpu().numpy()
        else:
            all_real_predictions = np.append(all_real_predictions, real_predictions.detach().cpu().numpy(), axis=0)
            all_fake_predictions = np.append(all_fake_predictions, fake_predictions.detach().cpu().numpy(), axis=0)
            all_labels = np.append(all_labels, inputs['classification_labels'].detach().cpu().numpy(), axis=0)

        num_steps += 1

    real_eval_loss = real_loss / num_steps
    fake_eval_loss= fake_loss / num_steps
    cum_real_predictions = np.argmax(all_real_predictions, axis=1)
    cum_fake_predictions = np.argmin(all_fake_predictions, axis=1)
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
        print('The {} is {}'.format(key, round(val, 3)))

    return -1


def main():
    class Args(object):
        def __init__(self):
            self.data_dir = '../ARC/ARC-with-context/'
            self.output_dir = 'output/'
            self.cache_dir = 'saved/'

            self.transformer_name = 'roberta'
            self.tokenizer_name = 'roberta-base'
            self.generator_model_type = 'roberta-reinforce'
            self.generator_model_name = '/home/kinne174/private/Output/transformers_gpu/language_modeling/saved/moon_roberta-base/'
            self.attention_model_type = 'essential-reinforce'

            self.classifier_model_type = 'roberta-reinforce'
            self.classifier_model_name = '/home/kinne174/private/Output/transformers_gpu/classification/saved/roberta-base/'

            self.discriminator_model_type = 'lstm-reinforce'
            self.discriminator_model_name = None
            self.discriminator_embedding_type = None
            self.discriminator_embedding_dim = 50
            self.discriminator_hidden_dim = 512
            self.discriminator_num_layers = 2
            self.discriminator_dropout = 0.10

            self.epochs = 3
            self.cutoff = None
            self.do_evaluate_test = False
            self.do_evaluate_dev = True
            self.use_gpu = True
            self.overwrite_output_dir = True
            self.overwrite_cache_dir = False
            self.seed = 1234
            self.max_length = 256
            self.do_lower_case = True
            self.essential_terms_hidden_dim = 512
            self.evaluate_all_models = False
            self.domain_words = ['moon', 'earth']

            self.evaluate_during_training = True
            self.learning_rate_classifier = 9e-5
            self.learning_rate_generator = 9e-6
            self.do_train = True
            self.batch_size = 2
            self.save_steps = 10
            self.essential_mu_p = 0.20
            self.use_corpus = True
            self.accumulation_steps = 25
            self.train_classifier = False
            self.train_discriminator = False

    args = Args()

    # Setup logging
    if not os.path.isdir('logging/{}'.format('_'.join(args.domain_words))):
        os.makedirs('logging/{}'.format('_'.join(args.domain_words)))
    num_logging_files = len(glob.glob('logging/{}/logging_g-{}_c-{}*'.format('_'.join(args.domain_words),
                                                                             args.generator_model_type,
                                                                             args.classifier_model_type)))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/{}/logging_g-{}_c-{}-{}'.format('_'.join(args.domain_words),
                                                                          args.generator_model_type,
                                                                          args.classifier_model_type,
                                                                          num_logging_files))

    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))

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

            assert -1 == evaluate(args, classifierM, generatorM, attentionM, tokenizer, cp)

if __name__ == '__main__':
    main()











