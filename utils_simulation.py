from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# general functions
def set_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

def detach_inputs(tuple_of_tensors):
    out_tensors = (t.detach() if hasattr(t, 'detach') else t for t in tuple_of_tensors)

    return out_tensors

tanh = nn.Tanh()
sigmoid = nn.Sigmoid()
crossentropy = nn.CrossEntropyLoss()
binarycrossentropywithlogits = nn.BCEWithLogitsLoss(reduction='none')
softmax = nn.Softmax(dim=-1)

# define models for generator, discriminator and classifier
# similar models to before
def attention(args, input_ids, attention_mask):
    masked_input_ids = input_ids.clone().detach()
    num_real_elements = torch.sum(attention_mask, dim=1, dtype=torch.int)
    my_attention_mask = torch.zeros_like(masked_input_ids)

    for i in range(input_ids.shape[0]):
        num_indices_to_mask = int(np.ceil(args.mu_p * num_real_elements[i].item()))

        indices_to_mask = np.random.permutation(num_real_elements[i].detach().cpu().item())[:num_indices_to_mask]

        for itm in indices_to_mask:
            itm += 1
            masked_input_ids[i, itm] = args.mask_id
            my_attention_mask[i, itm] = 1

    return masked_input_ids, my_attention_mask


def init_embedding(vocab_size, embedding_dim, do_random_weights=True, trainable=True):

    if not do_random_weights:
        embedding = nn.Embedding(vocab_size, vocab_size, padding_idx=0)
        bow_weights = torch.eye(vocab_size)
        embedding.load_state_dict({'weight': bow_weights})
        embedding_dim = vocab_size
    else:
        embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    if not trainable:
        embedding.weight.requires_grad = False

    return embedding, embedding_dim


class Seq2SeqReinforce(nn.Module):
    def __init__(self, config):
        super(Seq2SeqReinforce, self).__init__()
        self.embedding, self.embedding_dim = init_embedding(config.vocab_size, config.embedding_dim, not config.do_bow_weights, not config.do_fixed_embedding)

        self.embedding_to_encoder = nn.Sequential()
        self.embedding_to_encoder.add_module('linear_embedding_to_encoder', nn.Linear(self.embedding_dim, config.encoder_decoder_hidden_dim))
        self.embedding_to_encoder.add_module('relu_embedding_to_encoder', nn.ReLU())
        self.embedding_to_encoder.add_module('dropout_embedding_to_encoder', nn.Dropout(p=0.10))

        self.encoder = nn.LSTM(config.encoder_decoder_hidden_dim, config.encoder_decoder_hidden_dim,
                                                        batch_first=True, num_layers=config.num_layers,
                                                        dropout=0.10)

        self.embedding_to_decoder = nn.Sequential()
        self.embedding_to_decoder.add_module('linear_embedding_to_decoder', nn.Linear(self.embedding_dim, config.encoder_decoder_hidden_dim))
        self.embedding_to_decoder.add_module('relu_embedding_to_decoder', nn.ReLU())
        self.embedding_to_decoder.add_module('dropout_embedding_to_decoder', nn.Dropout(p=0.10))

        self.decoder = nn.LSTM(config.encoder_decoder_hidden_dim, config.encoder_decoder_hidden_dim,
                                                        batch_first=True, num_layers=config.num_layers,
                                                        dropout=0.10)

        self.classification_layer = nn.Linear(1, config.classification_layer_size, bias=True)

        self.fully_connected = nn.Sequential()
        self.fully_connected.add_module('linear1', nn.Linear(config.encoder_decoder_hidden_dim + config.classification_layer_size,
                                                             config.encoder_decoder_hidden_dim + config.classification_layer_size))
        self.fully_connected.add_module('relu', nn.ReLU())
        self.fully_connected.add_module('layernorm', nn.LayerNorm(config.encoder_decoder_hidden_dim + config.classification_layer_size))
        self.fully_connected.add_module('dropout', nn.Dropout(p=0.10))
        self.fully_connected.add_module('linear2', nn.Linear(config.encoder_decoder_hidden_dim + config.classification_layer_size, config.vocab_size))

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.mask_id = config.mask_id
        self.vocab_size = config.vocab_size

    def forward(self, input_ids, attention_mask, classification_labels):

        # input ids is [batch size, seq length]
        batch_size = input_ids.shape[0]

        # input ids is [batch size, seq length]
        # input_ids = input_ids.view(-1, input_ids.shape[-1])

        # embedded_ids is [batch size, seq length, embedding dim]
        embbeded_ids = self.embedding(input_ids)

        # embedded_ids is [batch size, seq length, encoder_decoder_hidden_dim]
        embbeded_ids = self.embedding_to_encoder(embbeded_ids)

        _, (hidden, cell) = self.encoder(embbeded_ids)

        # initial decoder input is [batch size, 1]
        decoder_input = input_ids[:, 0].unsqueeze(-1)

        output_prediction_scores = None

        for t in range(1, input_ids.shape[-1]):
            # embedded_input is [batch size, t, embedding dim]
            embedded_input = self.embedding(decoder_input)

            # embedded_input is [batch size, t, encoder decoder hidden dim]
            embedded_input = self.embedding_to_decoder(embedded_input)

            # decoder output is [batch size, 1, hidden dim]
            decoder_output, (hidden, cell) = self.decoder(embedded_input, (hidden, cell))

            # decoder output is [batch size, hidden dim]
            decoder_output = decoder_output[:, -1, :]

            # classifier states is [batch size, classifier layer size]
            classifier_states = self.classification_layer(classification_labels.view(-1, 1))

            # classifier states is [batch size, 4, classifier layer size]
            # classifier_states = classifier_states.view(-1, 4, classifier_states.shape[-1])

            # decoder classifier is [batch size, hidden dim + classifier layer size]
            decoder_classifier_output = torch.cat((decoder_output, classifier_states), dim=-1)

            # decoder prediction is [batch size, vocab size]
            decoder_prediction = self.fully_connected(decoder_classifier_output)

            # decoder prediction is [batch size, 1, vocab size]
            decoder_prediction = decoder_prediction.unsqueeze(1)

            decoder_prediction = self.log_softmax(decoder_prediction)

            cat = torch.distributions.Categorical(logits=decoder_prediction)
            sampled_ids = cat.sample()

            sampled_log_probs = cat.log_prob(sampled_ids)

            # output predictions is [batch size, t]
            if output_prediction_scores is None:
                output_prediction_scores = sampled_log_probs
            else:
                output_prediction_scores = torch.cat((output_prediction_scores, sampled_log_probs), dim=1)

            # next inputs is [batch size]
            next_inputs = torch.where(input_ids[:, t].unsqueeze(-1) == self.mask_id, sampled_ids, input_ids[:, t].unsqueeze(-1))

            # decoder input is [batch size, t+1]
            decoder_input = torch.cat((decoder_input, next_inputs), dim=-1)

            # attention mask is [batch size, sequence length]
            # if not torch.sum(attention_mask[:, t+1]):
            #     # no more relevant tokens
            #
            #     # output_prediction_scores is [batch size, seq length-1, vocab size]
            #     output_prediction_scores = torch.cat((output_prediction_scores, torch.zeros(batch_size, input_ids.shape[-1] - t - 1, self.vocab_size)), dim=-2)
            #
            #     break

        # output prediction scores is [batch size, seq length-1, vocab size]
        # output_prediction_scores = output_prediction_scores.view(-1, *output_prediction_scores.shape[1:])

        # output prediction scores is [batch size, seq length]
        output_prediction_scores = torch.cat((torch.zeros_like(output_prediction_scores[:, 0]).unsqueeze(-1), output_prediction_scores), dim=-1)

        # log softmax of scores
        # output_prediction_scores = self.log_softmax(output_prediction_scores)

        return output_prediction_scores, decoder_input


class ClassifierDiscriminatorReinforcement(nn.Module):
    def __init__(self, config):
        super(ClassifierDiscriminatorReinforcement, self).__init__()

        self.is_discriminator = config.is_discriminator

        self.embedding, self.embedding_dim = init_embedding(config.vocab_size, config.embedding_dim, not config.do_bow_weights, not config.do_fixed_embedding)

        self.linear1 = nn.Linear(self.embedding_dim, config.hidden_dim)
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim, batch_first=True, bidirectional=True,
                            num_layers=config.num_layers, dropout=0.10)
        self.linear2 = nn.Sequential()
        self.linear2.add_module('linear_fc1', nn.Linear(config.hidden_dim, config.fully_connected_dim))
        self.linear2.add_module('relu', nn.ReLU())
        self.linear2.add_module('layer norm', nn.LayerNorm(config.fully_connected_dim))
        self.linear2.add_module('dropout', nn.Dropout(p=0.10))
        self.linear2.add_module('linear_fc2', nn.Linear(config.fully_connected_dim, 1))

    def forward(self, input_ids):

        # input_ids is [batch size, seq length]

        # word_embeddings is [batch size, seq length, embedding dim]
        word_embeddings = self.embedding(input_ids)

        # transformed word embeddings is [batch size, seq length, lstm hidden dim]
        transformed_word_embeddings = self.linear1(word_embeddings)

        # all_hidden is [batch size, seq length, bidirectional*lstm hidden dim]
        # last hidden is [bidirectional*num layers, batch size, lstm hidden dim]
        all_hidden, (last_hidden, last_cell) = self.lstm(transformed_word_embeddings)

        bidirectional_dim = 2 if self.lstm.bidirectional else 1
        num_layers_dim = self.lstm.num_layers

        if self.is_discriminator:
            # lstm out is [batch size, seq length, lstm hidden dim]
            lstm_out = torch.mean(all_hidden.view(input_ids.shape[0], input_ids.shape[1], bidirectional_dim, -1), dim=2)
        else:
            # lstm out is [batch size, lstm hidden dim]
            lstm_out = torch.mean(last_hidden.view(num_layers_dim, bidirectional_dim, input_ids.shape[0], -1)[-1, ...], dim=0)

        # logits is [batch size, seq length, 1]
        logits = self.linear2(lstm_out)

        return logits.squeeze()

# classifier can just be another discriminator
class Config(object):
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


# initialize the models for each fold
def initialize_models(args):
    logger.info('Initializing config dicts')
    generator_config = Config({'vocab_size': args.vocab_size,
                               'embedding_dim': args.embedding_dim,
                               'do_bow_weights': args.do_bow_weights,
                               'do_fixed_embedding': args.do_fixed_embedding,
                               'encoder_decoder_hidden_dim': args.generator_hidden_dim,
                               'num_layers': args.generator_num_layers,
                               'classification_layer_size': args.generator_classification_layer_size,
                               'mask_id': args.mask_id})
    classifier_config = Config({'vocab_size': args.vocab_size,
                                'embedding_dim': args.embedding_dim,
                                'hidden_dim': args.classifier_discriminator_hidden_dim,
                                'num_layers': args.classifier_discriminator_num_layers,
                                'fully_connected_dim': args.classifier_discriminator_fc_dim,
                                'do_bow_weights': args.do_bow_weights,
                                'do_fixed_embedding': args.do_fixed_embedding,
                                'is_discriminator': False})
    discriminator_config = Config({'vocab_size': args.vocab_size,
                                'embedding_dim': args.embedding_dim,
                                'hidden_dim': args.classifier_discriminator_hidden_dim,
                                'num_layers': args.classifier_discriminator_num_layers,
                                'fully_connected_dim': args.classifier_discriminator_fc_dim,
                                'do_bow_weights': args.do_bow_weights,
                                'do_fixed_embedding': args.do_fixed_embedding,
                                'is_discriminator': True})

    logger.info('Initializing models')
    generatorM = Seq2SeqReinforce(generator_config)
    discriminatorM = ClassifierDiscriminatorReinforcement(discriminator_config)
    classifierM = ClassifierDiscriminatorReinforcement(classifier_config)

    logger.info('Loading models to {}'.format(args.device))
    generatorM.to(args.device)
    discriminatorM.to(args.device)
    classifierM.to(args.device)

    return generatorM, discriminatorM, classifierM


# define data loading/ generating based on simulation number
def load_features(args):
    input_ids = []
    attention_mask = []
    classifier_labels = None
    # keep 0 and 1 protected for padding id and mask id respectively
    if args.simulation_number == 1:
        # generate counting data starting from 2 and going up to some random number
        start = 2
        stop = np.random.randint(low=start, high=args.max_length - start + 1, size=args.num_observations)
        for s in stop:
            input_ids.append(torch.tensor([0] + list(range(start, s+1)) + [0]*(args.max_length - (s - start + 1) - 1)).unsqueeze(0))
            attention_mask.append(torch.tensor([0] + [1]*(s+1 - start) + [0]*(args.max_length - (s - start + 1) - 1)).unsqueeze(0))

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        classifier_labels = torch.zeros(args.num_observations, dtype=torch.float)

    elif args.simulation_number == 2:
        # generate counting data starting after mask id for random amounts depending on max length and vocab size
        break_flag = False
        while True:
            stop_arr = np.random.randint(low=2, high=args.vocab_size, size=args.num_observations)
            start_arr = np.random.randint(low=np.maximum(2, stop_arr+1 - args.max_length), high=stop_arr+1, size=args.num_observations)
            for stop, start in zip(stop_arr, start_arr):
                if stop - start <= 1 / args.mu_p:
                    continue
                input_ids.append(torch.tensor([0] + list(range(start, stop)) + [0]*(args.max_length - (stop - start) - 1)).unsqueeze(0))
                attention_mask.append(torch.tensor([0] + [1]*(stop - start) + [0]*(args.max_length - (stop - start) - 1)).unsqueeze(0))

                if len(input_ids) == args.num_observations:
                    break_flag = True
                    break

            if break_flag:
                break

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        classifier_labels = torch.zeros(args.num_observations, dtype=torch.float)

    elif args.simulation_number == 3:
        # generate even and odd data
        integers = np.random.randint(low=1, high=args.vocab_size//2, size=(args.num_observations, args.max_length)) * 2
        even_odd_flag = np.random.randint(low=0, high=2, size=args.num_observations)
        lengths = np.random.randint(low=10, high=args.max_length, size=args.num_observations)

        for i in range(args.num_observations):
            integers_to_add = integers[i, :lengths[i]]
            if even_odd_flag[i] == 1:
                integers_to_add += 1
            input_ids.append(torch.tensor([0] + integers_to_add.tolist() + [0]*(args.max_length - lengths[i] - 1)).unsqueeze(0))
            attention_mask.append(torch.tensor([0] + [1]*lengths[i] + [0]*(args.max_length - lengths[i] - 1)).unsqueeze(0))

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        classifier_labels = torch.zeros(args.num_observations, dtype=torch.float)

    elif args.simulation_number == 4:
        # generate even and odd data with caveat of fixed number for sum, assign class labels accordingly
        raise NotImplementedError
    else:
        raise NotImplementedError

    assert input_ids.shape[0] == attention_mask.shape[0] == classifier_labels.shape[0] == args.num_observations
    assert input_ids.shape[1] == attention_mask.shape[1] == args.max_length

    dataset = TensorDataset(input_ids, attention_mask, classifier_labels)

    return dataset
# functions to train generator and train classifier/ discriminator
def train_generator(args, masked_input_ids, real_input_ids, my_attention_mask, attention_mask,
                    classification_labels, generatorM, discriminatorM, classifierM, baseline):
    fake_logprobs, fake_input_ids = generatorM(masked_input_ids, attention_mask, classification_labels)

    with torch.no_grad():
        logits_discriminator = discriminatorM(fake_input_ids)
        if args.train_discriminator:
            rewards_discriminator = 2. * sigmoid(logits_discriminator) - 1.
        else:
            if args.simulation_number in [1, 2]:
                rewards_discriminator = 2. * torch.eq(real_input_ids, fake_input_ids) - 1.
            elif args.simulation_number in [3]:
                odd_even_indicator = torch.min(torch.ones_like(real_input_ids[:, 0]),
                                               torch.sum(real_input_ids % 2, dim=-1)).unsqueeze(-1)
                rewards_discriminator = 2. * torch.eq(odd_even_indicator, fake_input_ids % 2) - 1

        logits_fake_classifier = classifierM(fake_input_ids)
        if args.train_classifier:
            rewards_classifier = 2. * (
                        classification_labels * sigmoid(logits_fake_classifier) + (1 - classification_labels) * (
                            1 - sigmoid(logits_fake_classifier))) - 1.
        else:
            rewards_classifier = 2. * torch.eq(classification_labels, 1. * (torch.sum(fake_input_ids, dim=-1) >= args.fixed_point)) - 1

    rewards_discriminator *= (1 - args.rewards_decay)
    rewards_classifier *= args.rewards_decay

    # calculate rewards
    gamma_tensor = args.gamma ** torch.arange(fake_input_ids.shape[-1], dtype=torch.float).to(args.device)

    rewards = (rewards_discriminator + rewards_classifier.unsqueeze(-1)) * my_attention_mask
    # rewards = rewards.view(-1, rewards.shape[-1])
    # my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])

    # rewards = rewards[my_attention_mask.nonzero(as_tuple=True)]

    rewards_gamma = torch.empty_like(rewards).to(args.device)
    for j in range(rewards_gamma.shape[-1]):
        rewards_gamma[:, j] = torch.sum(gamma_tensor[:len(gamma_tensor) - j].unsqueeze(0) * rewards[:, j:], dim=1)

    baseline['num_steps'] += rewards.shape[0]
    baseline['total_mean_rewards'] += torch.sum(
        torch.sum(rewards_gamma * my_attention_mask, dim=-1) / (torch.sum(my_attention_mask, dim=-1) + 1e-6))

    new_baseline = baseline['prev'] * args.lambda_baseline + (1 - args.lambda_baseline) * (
                baseline['total_mean_rewards'] / baseline['num_steps'])

    rewards_minus_baseline = rewards_gamma - new_baseline

    log_vocab_probs = fake_logprobs.view(-1, fake_logprobs.shape[-1]) * my_attention_mask
    out_rewards = torch.mean(torch.sum(-1 * rewards_minus_baseline * log_vocab_probs, dim=-1))

    baseline['prev'] = new_baseline

    return out_rewards, fake_input_ids, (logits_discriminator, logits_fake_classifier)

def train_classifier_discriminator(args, fake_input_ids, my_attention_mask, real_input_ids, attention_mask,
                                   classification_labels, discriminatorM, classifierM):
    logits_real_c = classifierM(real_input_ids)
    error_real_c = torch.mean(binarycrossentropywithlogits(logits_real_c, classification_labels))

    logits_fake_c = classifierM(fake_input_ids)
    error_fake_c = -1 * torch.mean(binarycrossentropywithlogits(logits_fake_c, classification_labels))

    logits_real_d = discriminatorM(real_input_ids)
    error_real_d = binarycrossentropywithlogits(logits_real_d, my_attention_mask.float()) # training to denote 1 as real
    error_real_d = torch.sum(error_real_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    logits_fake_d = discriminatorM(fake_input_ids)
    error_fake_d = -1 * binarycrossentropywithlogits(logits_fake_d, my_attention_mask.float()) # training to denote 0 as fake
    error_fake_d = torch.sum(error_fake_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    return {'logits_real_c': logits_real_c, 'logits_fake_c': logits_fake_c, 'logits_real_d': logits_real_d,
            'logits_fake_d': logits_fake_d}, \
           {'error_real_c': error_real_c, 'error_fake_c': error_fake_c, 'error_real_d': error_real_d,
            'error_fake_d': error_fake_d}

# simulation specific accuracy calculators
def simulation_one(fake_input_ids, real_input_ids, my_attention_mask):
    # should be counting from two, accuracy is percent correct
    mask_indices = my_attention_mask.nonzero(as_tuple=True)
    num_correct = torch.sum(torch.eq(fake_input_ids[mask_indices], real_input_ids[mask_indices])).item()

    return num_correct / float(len(mask_indices[0]))

def simulation_two(fake_input_ids, real_input_ids, my_attention_mask):
    # accuracy is percent correct
    mask_indices = my_attention_mask.nonzero(as_tuple=True)
    num_correct = torch.sum(torch.eq(fake_input_ids[mask_indices], real_input_ids[mask_indices])).item()

    return num_correct / float(len(mask_indices[0]))

def simulation_three(fake_input_ids, real_input_ids, my_attention_mask):
    odd_even_indicator = torch.min(torch.ones_like(real_input_ids[:, 0]), torch.sum(real_input_ids % 2, dim=-1)).unsqueeze(-1)
    all_num_same = torch.eq(odd_even_indicator, fake_input_ids % 2)
    num_correct = torch.sum(all_num_same[my_attention_mask.nonzero(as_tuple=True)]).item()

    return num_correct / float(my_attention_mask.nonzero().shape[0])


# do k-fold cross validation over some number of epochs for each fold
# k-fold done in main, just train here
def train(args, dataset, generatorM, discriminatorM, classifierM,):

    # use pytorch data loaders to cycle through the data
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    # optimizers
    classifierO = optim.Adam(classifierM.parameters(), lr=args.learning_rate_classifier, weight_decay=args.weight_decay_classifier)
    generatorO = optim.Adam(generatorM.parameters(), lr=args.learning_rate_generator, weight_decay=args.weight_decay_generator)
    discriminatorO = optim.Adam(discriminatorM.parameters(), lr=args.learning_rate_discriminator, weight_decay=args.weight_decay_discriminator)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    update_step = 0
    global_step = 0

    # zero out gradient of networks
    generatorM.zero_grad()
    classifierM.zero_grad()
    discriminatorM.zero_grad()

    # gradient parameters for classifier and discriminator
    classifier_gradients = []
    discriminator_gradients = []

    accumulated_error_real_d = []
    accumulated_error_fake_d = []

    accumulated_error_generator = []
    accumulated_rewards_generator = []

    accumulated_error_fake_c = []
    accumulated_error_real_c = []

    results = {}
    results['real_loss_classifier'] = []
    results['real_loss_discriminator'] = []
    results['fake_loss_classifier'] = []
    results['fake_loss_discriminator'] = []
    results['real_accuracy_classifier'] = []
    results['fake_accuracy_classifier'] = []
    results['fake_accuracy_discriminator'] = []
    results['real_accuracy_discriminator'] = []
    results['fake_accuracy_generator'] = []

    logger.info('Starting to train!')
    logger.info('There are {} training examples.'.format(len(dataset)))

    for epoch, _ in enumerate(train_iterator):

        # training tracking
        num_discriminator_seen, num_classification_seen = 0, 0
        num_training_correct_real_classifier = 0
        num_training_correct_real_discriminator, num_training_correct_fake_discriminator = 0, 0
        num_training_correct_generator_classifier, num_training_correct_generator_discriminator = 0, 0

        # baseline dict: (prev baseline, total reward, num_steps)
        baseline_dict = {'prev': 0, 'total_mean_rewards': 0, 'num_steps': 0}

        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))

        for batch_iterate, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            real_inputs = {'input_ids': batch[0],
                           'attention_mask': batch[1],
                           'classification_labels': batch[2],
                           }

            # Train generator
            generatorM.train()

            # Train classifier/ discriminator
            classifierM.train()
            discriminatorM.train()

            # get the masked input ids
            masked_input_ids, my_attention_mask = attention(args, real_inputs['input_ids'], real_inputs['attention_mask'])

            assert masked_input_ids.device == my_attention_mask.device == args.device

            # get rewards for generator
            generator_rewards, fake_input_ids, (logits_gen_d, logits_gen_c) = train_generator(args,
                                                                                           masked_input_ids,
                                                                                           real_inputs['input_ids'],
                                                                                           my_attention_mask,
                                                                                           real_inputs['attention_mask'],
                                                                                           real_inputs['classification_labels'],
                                                                                           generatorM,
                                                                                           discriminatorM,
                                                                                           classifierM,
                                                                                           baseline_dict)

            if args.train_generator:
                generator_rewards /= args.accumulation_steps
                generator_rewards.backward()

            accumulated_rewards_generator.append(generator_rewards.detach().item())

            num_training_correct_generator_discriminator += int(torch.sum(
                torch.eq(my_attention_mask, tanh(logits_gen_d).sign())[
                    my_attention_mask.nonzero(as_tuple=True)], dtype=torch.int).detach().cpu())

            if args.simulation_number == 1:
                accumulated_error_generator.append(1. - simulation_one(fake_input_ids, real_inputs['input_ids'], my_attention_mask))
            elif args.simulation_number == 2:
                accumulated_error_generator.append(1. - simulation_two(fake_input_ids, real_inputs['input_ids'], my_attention_mask))
            elif args.simulation_number == 3:
                accumulated_error_generator.append(1. - simulation_three(fake_input_ids, real_inputs['input_ids'], my_attention_mask))

            # zero out classifier and discriminator so they do not accumulate these gradients
            classifierM.zero_grad()
            discriminatorM.zero_grad()

            classifierM.train()
            discriminatorM.train()

            # detach the inputs so the gradient graphs don't reach back, only need them for discriminator/ classifier
            fake_input_ids, my_attention_mask = detach_inputs([fake_input_ids, my_attention_mask])

            logits, errors = train_classifier_discriminator(args, fake_input_ids, my_attention_mask,
                                                            real_inputs['input_ids'],
                                                            real_inputs['attention_mask'],
                                                            real_inputs['classification_labels'],
                                                            discriminatorM, classifierM)

            # error_real_c is classification error, error_real_d is discriminator error
            if errors['error_real_c'] is not None:
                errors['error_real_c'] *= (1 - args.classification_decay)
                errors['error_real_c'] /= args.accumulation_steps * args.classifier_hop_steps
                errors['error_real_c'].backward()

                accumulated_error_real_c.append(errors['error_real_c'].detach().item())

                errors['error_fake_c'] *= args.classification_decay
                errors['error_fake_c'] /= args.accumulation_steps * args.classifier_hop_steps
                errors['error_fake_c'].backward()

                accumulated_error_fake_c.append(errors['error_fake_c'].detach().item())

                num_classification_seen += real_inputs['input_ids'].shape[0]
                num_training_correct_real_classifier += torch.sum(
                    torch.eq(real_inputs['classification_labels'],
                             sigmoid(logits['logits_real_c']).round()), dtype=torch.int).detach().cpu()
                num_training_correct_generator_classifier += torch.sum(
                    torch.eq(real_inputs['classification_labels'],
                    sigmoid(logits_gen_c).round()), dtype=torch.int).detach().cpu()

            num_discriminator_seen += int(my_attention_mask.nonzero().shape[0])

            if errors['error_real_d'] is not None:
                errors['error_real_d'] /= args.accumulation_steps * args.discriminator_hop_steps
                errors['error_real_d'].backward()

                accumulated_error_real_d.append(errors['error_real_d'].detach().item())

                num_training_correct_real_discriminator += torch.sum(
                    torch.eq(my_attention_mask, tanh(logits['logits_real_d']).sign())[
                        my_attention_mask.nonzero(as_tuple=True)], dtype=torch.int).detach().cpu()

            if errors['error_fake_d'] is not None:
                errors['error_fake_d'] /= args.accumulation_steps * args.discriminator_hop_steps
                errors['error_fake_d'].backward()

                accumulated_error_fake_d.append(errors['error_fake_d'].detach().item())

                num_training_correct_fake_discriminator += torch.sum(
                    torch.eq(-1 * my_attention_mask, tanh(logits['logits_fake_d']).sign())[
                        my_attention_mask.nonzero(as_tuple=True)], dtype=torch.int).detach().cpu()

            # save gradient parameters in a list
            # for classifier
            if args.train_classifier:
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
                rewards_generator_std, rewards_generator_mean = torch.std_mean(torch.tensor(accumulated_rewards_generator))
                error_real_c_std, error_real_c_mean = torch.std_mean(torch.tensor(accumulated_error_real_c))
                error_fake_c_std, error_fake_c_mean = torch.std_mean(torch.tensor(accumulated_error_fake_c))
                error_real_d_std, error_real_d_mean = torch.std_mean(torch.tensor(accumulated_error_real_d))
                error_fake_d_std, error_fake_d_mean = torch.std_mean(torch.tensor(accumulated_error_fake_d))
                error_generator_std, error_generator_mean = torch.std_mean(torch.tensor(accumulated_error_generator))

                messages = {}
                messages[
                    'message_generator_batch_rewards'] = 'The batch Generator rewards mean: {:.3f} and std: {:.3f}'.format(
                    rewards_generator_mean, rewards_generator_std)

                messages['message_generator_batch_error'] = 'The batch Generator errors mean: {:.3f} and std: {:.3f}'.format(
                    error_generator_mean, error_generator_std)

                messages['message_classifier_batch_error'] = 'The batch Classifier mean (std) errors: real {:.3f} ({:.3f}) + fake {:.3f} ({:.3f}) = {:.3f} '.format(
                                                                                    error_real_c_mean,
                                                                                    error_real_c_std,
                                                                                    error_fake_c_mean,
                                                                                    error_fake_c_std,
                                                                                    error_real_c_mean + error_fake_c_mean)

                messages[
                    'message_discriminator_batch_error'] = 'The batch Discriminator mean (std) errors: real {:.3f} ({:.3f}) + fake {:.3f} ({:.3f}) = {:.3f}'.format(
                    error_real_d_mean,
                    error_real_d_std,
                    error_fake_d_mean,
                    error_fake_d_std,
                    error_real_d_mean + error_fake_d_mean)


                messages['message_classifier_real_running_total_correct'] = '\treal Classification is {} out of {} for a percentage of {:.3f}'.format(
                        num_training_correct_real_classifier, num_classification_seen,
                        num_training_correct_real_classifier / float(num_classification_seen))

                messages['message_classifier_generator_running_total_correct'] = '\tGenerator Classification is {} out of {} for a percentage of {:.3f}'.format(
                        num_training_correct_generator_classifier, num_classification_seen,
                        num_training_correct_generator_classifier / float(num_classification_seen))

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

                    relevant_messages = []
                    if args.train_generator:
                        relevant_messages.append('message_generator_batch_error')
                        relevant_messages.append('message_generator_batch_rewards')
                    if args.train_discriminator:
                        relevant_messages.append('message_discriminator_batch_error')
                        relevant_messages.append('message_generator_discriminator_running_total_correct')
                        relevant_messages.append('message_discriminator_real_running_total_correct')
                        relevant_messages.append('message_dicriminator_fake_running_total_correct')
                    if args.train_classifier:
                        relevant_messages.append('message_classifier_batch_error')
                        relevant_messages.append('message_classifier_real_running_total_correct')
                        relevant_messages.append('message_classifier_generator_running_total_correct')

                    for title, m in messages.items():
                        if title in relevant_messages:
                            logger.info(m)
                    logger.info('Example: {}'.format(','.join([('*' if bool(my_attention_mask[0, i]) else '') +
                                                               str(fake_input_ids[0, i].item()) +
                                                               ('*' if bool(my_attention_mask[0, i]) else '')
                                                               for i in range(fake_input_ids.shape[1]) if real_inputs['attention_mask'][0, i] == 1])))


                results['real_loss_classifier'].append(error_real_c_mean)
                results['real_loss_discriminator'].append(error_real_d_mean)
                results['fake_loss_classifier'].append(error_fake_c_mean)
                results['fake_loss_discriminator'].append(error_fake_d_mean)
                results['real_accuracy_classifier'].append(num_training_correct_real_classifier / float(num_classification_seen))
                results['fake_accuracy_classifier'].append(num_training_correct_generator_classifier / float(num_classification_seen))
                results['fake_accuracy_discriminator'].append(num_training_correct_generator_discriminator / float(num_discriminator_seen))
                results['real_accuracy_discriminator'].append((num_training_correct_real_discriminator + num_training_correct_fake_discriminator) / 2*float(num_discriminator_seen))
                results['fake_accuracy_generator'].append(error_generator_mean)

                accumulated_error_real_d = []
                accumulated_error_fake_d = []

                accumulated_error_generator = []
                accumulated_rewards_generator = []

                accumulated_error_real_c = []
                accumulated_error_fake_c = []

                if args.train_generator:
                    generatorO.step()

                # update classifier/discriminator parameters
                if (update_step + 1) % args.classifier_hop_steps == 0:
                    if args.train_classifier:
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
            global_step += 1
        epoch_iterator.close()
    train_iterator.close()

    return generatorM, discriminatorM, classifierM, results

# TODO for different simulation numbers automate rewards and whether to train the classifier/discriminator

def test(args, dataset, generatorM, discriminatorM, classifierM):
    results = {}

    # change models to eval
    classifierM.eval()
    generatorM.eval()
    discriminatorM.eval()

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    logger.info('Starting Evaluation!')
    logger.info('Number of examples: {}'.format(len(dataset)))

    real_loss_classifier = 0.
    real_loss_discriminator = 0.
    fake_loss_classifier = 0.
    fake_loss_discriminator = 0.


    real_predictions_classifier = None
    real_predictions_discriminator = None
    fake_predictions_classifier = None
    fake_predictions_discriminator = None

    num_steps = 0

    num_batches = len(eval_dataloader)
    with torch.no_grad():
        for batch_ind, batch in tqdm(enumerate(eval_dataloader),
                                     'Evaluating {} batches of batch size {}'.format(num_batches,
                                                                                     args.batch_size)):
            batch = tuple(t.to(args.device) for t in batch)
            real_inputs = {'input_ids': batch[0],
                           'attention_mask': batch[1],
                           'classification_labels': batch[2],
                           }

            masked_input_ids, my_attention_mask = attention(args, real_inputs['input_ids'],
                                                            real_inputs['attention_mask'])

            _, fake_input_ids = generatorM(masked_input_ids,
                                           real_inputs['attention_mask'],
                                           real_inputs['classification_labels'])

            fake_logits_classifier = classifierM(fake_input_ids)
            fake_logits_discriminator = discriminatorM(fake_input_ids)

            real_logits_classifier = classifierM(real_inputs['input_ids'])
            real_logits_discriminator = discriminatorM(real_inputs['input_ids'])

            fake_error_classifier = -1 * torch.mean(binarycrossentropywithlogits(fake_logits_classifier, real_inputs['classification_labels']))
            fake_error_discriminator = -1 * binarycrossentropywithlogits(fake_logits_discriminator, my_attention_mask.float())
            fake_error_discriminator = torch.sum(fake_error_discriminator[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

            real_error_classifier = torch.mean(binarycrossentropywithlogits(real_logits_classifier, real_inputs['classification_labels']))
            real_error_discriminator = binarycrossentropywithlogits(real_logits_discriminator, my_attention_mask.float())
            real_error_discriminator = torch.sum(real_error_discriminator[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

            fake_loss_classifier += fake_error_classifier.detach().cpu().item()
            fake_loss_discriminator += fake_error_discriminator.detach().cpu().item()

            real_loss_classifier += real_error_classifier.detach().cpu().item()
            real_loss_discriminator += real_error_discriminator.detach().cpu().item()

            if real_predictions_classifier is None:
                real_predictions_classifier = sigmoid(real_logits_classifier).round().detach().cpu()
                real_predictions_discriminator = sigmoid(real_logits_discriminator).round().detach().cpu()

                fake_predictions_classifier = sigmoid(fake_logits_classifier).round().detach().cpu()
                fake_predictions_discriminator = sigmoid(fake_logits_discriminator).round().detach().cpu()

                all_my_attention_mask = my_attention_mask

            else:
                real_predictions_classifier = torch.cat((real_predictions_classifier, sigmoid(real_logits_classifier).round().detach().cpu()), dim=0)
                real_predictions_discriminator = torch.cat((real_predictions_discriminator, sigmoid(real_logits_discriminator).round().detach().cpu()), dim=0)

                fake_predictions_classifier = torch.cat((fake_predictions_classifier, sigmoid(fake_logits_classifier).round().detach().cpu()), dim=0)
                fake_predictions_discriminator = torch.cat((fake_predictions_discriminator, sigmoid(fake_logits_discriminator).round().detach().cpu()), dim=0)

                all_my_attention_mask = torch.cat((all_my_attention_mask, my_attention_mask), dim=0)

            if args.simulation_number == 1:
                fake_accuracy_generator = simulation_one(fake_input_ids, real_inputs['input_ids'], my_attention_mask)
            elif args.simulation_number == 2:
                fake_accuracy_generator = simulation_two(fake_input_ids, real_inputs['input_ids'], my_attention_mask)
            elif args.simulation_number == 3:
                fake_accuracy_generator = simulation_three(fake_input_ids, real_inputs['input_ids'], my_attention_mask)

            num_steps += real_inputs['input_ids'].shape[0]

    fake_loss_classifier /= num_steps
    fake_loss_discriminator /= num_steps

    real_loss_classifier /= num_steps
    real_loss_discriminator /= num_steps

    fake_acuracy_classifier = accuracy_score(dataset.tensors[2].int(), fake_predictions_classifier.int())
    fake_accuracy_discriminator = accuracy_score(torch.ones(all_my_attention_mask.nonzero().shape[0]), fake_predictions_discriminator[all_my_attention_mask.nonzero(as_tuple=True)])

    real_accuracy_classifier = accuracy_score(dataset.tensors[2].int(), real_predictions_classifier.int())
    real_accuracy_discriminator = accuracy_score(torch.zeros(all_my_attention_mask.nonzero().shape[0]), real_predictions_discriminator[all_my_attention_mask.nonzero(as_tuple=True)])

    results['fake_loss_classifier'] = fake_loss_classifier
    results['fake_loss_discriminator'] = fake_loss_discriminator
    results['fake_accuracy_classifier'] = fake_acuracy_classifier
    results['fake_accuarcy_discriminator'] = fake_accuracy_discriminator
    results['real_loss_classifier'] = real_loss_classifier
    results['real_loss_discriminator'] = real_loss_discriminator
    results['real_accuracy_classifier'] = real_accuracy_classifier
    results['real_accuracy_discriminator'] = real_accuracy_discriminator
    results['fake_accuracy_generator'] = fake_accuracy_generator

    logger.info('After evaluating')
    for key, val in results.items():
        logger.info('The {} is {}'.format(key, round(val, 3)))

    return results


# output plots of training and example/ expected output
def output_results(results_train, results_test):
    pass

# main with arguments
def main():
    class Args(object):
        def __init__(self):
            # total args
            self.simulation_number = 3
            self.seed = 1234
            self.cuda_num = 6
            self.num_splits = 3
            self.mask_id = 1
            self.quiet = False

            # dataset args
            self.num_observations = 10000
            self.max_length = 50
            self.vocab_size = 75
            self.fixed_point = 50

            # general model args
            self.do_bow_weights = False
            self.do_fixed_embedding = False
            self.embedding_dim = 50

            # generator args
            self.generator_hidden_dim = 512
            self.generator_num_layers = 2
            self.generator_classification_layer_size = 1

            # classifier/ discriminator args
            self.classifier_discriminator_hidden_dim = 256
            self.classifier_discriminator_num_layers = 2
            self.classifier_discriminator_fc_dim = 128

            # training args
            self.batch_size = 100
            self.accumulation_steps = 1
            self.epochs = 250
            self.train_generator = True
            self.train_discriminator = True
            self.discriminator_hop_steps = 1
            self.classifier_hop_steps = 1
            self.mu_p = 0.10
            self.rewards_decay = 0.
            self.classification_decay = 0.5

            # hyperparameters
            self.learning_rate_classifier = 9e-5
            self.learning_rate_generator = 9.5e-6
            self.learning_rate_discriminator = 9.5e-5
            self.weight_decay_classifier = 1
            self.weight_decay_discriminator = 1
            self.weight_decay_generator = 0
            self.gamma = 0.70
            self.lambda_baseline = 0.15

            # based on simulation no
            if self.simulation_number in [1, 2, 3]:
                self.train_classifier = False
                self.rewards_decay = 0.
            else:
                self.train_classifier = True

    args = Args()

    # Setup logging
    if not os.path.isdir('logging/simulation'):
        os.makedirs('logging/simulation')
    num_logging_files = len(glob.glob('logging/simulation/logging_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler('logging/simulation/logging_{}'.format(
                                                      num_logging_files)),
                                  logging.StreamHandler()])

    # set seed
    set_seed(args)

    # print out arguments
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    # get whether running on cpu or gpu
    device = torch.device('cuda:{}'.format(args.cuda_num))
    args.device = device
    logger.info('Using device {}'.format(args.device))

    # get simulation dataset
    dataset = load_features(args)

    kf = KFold(n_splits=args.num_splits)

    results_test = []
    results_train = []

    for train_index, test_index in kf.split(dataset.tensors[0]) :

        train_input_ids = dataset.tensors[0][train_index]
        train_attention_mask = dataset.tensors[1][train_index]
        train_classifier_labels = dataset.tensors[2][train_index]

        test_input_ids = dataset.tensors[0][test_index]
        test_attention_mask = dataset.tensors[1][test_index]
        test_classifier_labels = dataset.tensors[2][test_index]

        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_classifier_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_classifier_labels)

        # initialize and return models
        generatorM, discriminatorM, classifierM = initialize_models(args)

        # train models
        generatorM, discriminatorM, classifierM, results = train(args, train_dataset, generatorM, discriminatorM, classifierM)
        results_train.append(results)

        results = test(args, test_dataset, generatorM, discriminatorM, classifierM)

        results_test.append(results)

        # output_results(results_train, results_test)


if __name__ == '__main__':
    main()

