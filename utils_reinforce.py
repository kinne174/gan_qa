import torch
import torch.nn as nn
import os
import logging
import numpy as np
import json
import random

logger = logging.getLogger(__name__)

sigmoid = nn.Sigmoid()

def train_generator(args, input_ids, attention_mask, my_attention_mask, token_type_ids, classification_labels,
                    sentences_type, generatorM, classifierM, discriminatorM):

    generatorM.train()

    # input_ids = fake_inputs['input_ids']
    # attention_mask = fake_inputs['attention_mask']
    # my_attention_mask = fake_inputs['my_attention_mask']
    # token_type_ids = fake_inputs['token_type_ids']
    # classification_labels = fake_inputs['classification_labels']
    # sentences_type = fake_inputs['sentences_type']

    # draw fake probability for words

    fake_vocabulary_probs = generatorM(input_ids, attention_mask, token_type_ids)
    # fake_vocabulary_probs = fake_vocabulary_probs.view(-1, fake_vocabulary_probs.shape[2], fake_vocabulary_probs.shape[3])

    # choose words at random or argmax or based on probability
    fake_action = word_index_selector(fake_vocabulary_probs, args.reinforce_action_method)
    fake_input_ids = (my_attention_mask*fake_action + (1 - my_attention_mask)*input_ids).long()
    # throw fake_action to device
    fake_action = fake_action.to(args.device)

    # score words with discriminator and classifier
    logits_discriminator = discriminatorM(fake_input_ids)
    logits_classifier = classifierM(fake_input_ids, attention_mask, token_type_ids)

    rewards_discriminator = 2*sigmoid(logits_discriminator) - 1
    rewards_classifier = my_CrossEntropyLoss(logits_classifier, classification_labels.nonzero()[:, 1].long(), neg_one=False)

    # calculate errors
    # have to make sure to use fake_vocabulary_probs to get connection to generator, see eq 9 in scratch gan
    loss = discount_rewards(args, fake_vocabulary_probs, fake_action, my_attention_mask,
                            rewards_discriminator, sentences_type, rewards_classifier)

    return loss, fake_input_ids, (logits_discriminator, logits_classifier)


def train_classifier_discriminator(args, fake_input_ids, real_input_ids, attention_mask, my_attention_mask,
                                   token_type_ids, classification_labels, sentences_type, discriminatorM, classifierM):

    # fake_input_ids = fake_inputs['input_ids']
    # real_input_ids = real_inputs['input_ids']
    # attention_mask = fake_inputs['attention_mask']
    # my_attention_mask = fake_inputs['my_attention_mask']
    # token_type_ids = fake_inputs['token_type_ids']
    # classification_labels = real_inputs['classification_labels']
    # sentences_type = fake_inputs['sentences_type']

    if sentences_type.nonzero().shape[0]:
        logits_real_c = classifierM(real_input_ids, attention_mask, token_type_ids)
        error_real_c = my_CrossEntropyLoss(logits_real_c, classification_labels.nonzero()[:, 1].long(), neg_one=True)
        error_real_c = torch.sum(sentences_type * error_real_c) / torch.sum(sentences_type)
    else:
        logits_real_c, error_real_c = None, None

    logits_real_d = discriminatorM(real_input_ids)
    # error_real_d = Wloss(logits_real_d, real_discriminator_labels, my_attention_mask, use_tanh=True)
    error_real_d = my_BinaryCrossEntropyLoss(logits_real_d, my_attention_mask, neg_one=True)
    error_real_d = torch.sum(error_real_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    logits_fake_d = discriminatorM(fake_input_ids)
    # error_fake_d = Wloss(logits_fake_d, fake_discriminator_labels, my_attention_mask, use_tanh=True)
    error_fake_d = my_BinaryCrossEntropyLoss(logits_fake_d, my_attention_mask, neg_one=False)
    error_fake_d = torch.sum(error_fake_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    return (logits_real_c, logits_real_d, logits_fake_d), (error_real_c, error_real_d, error_fake_d)


def word_index_selector(vocab_probs, method):
    # TODO should I be implenting an annealing completely random choice here?
    if method == 'argmax':
        # do argmax over the third dimension of the probabilities
        out = torch.argmax(vocab_probs, -1)
        assert out.shape == vocab_probs.shape[:-1], 'shape of outputted action should match [batch * 4 * max_length]'
        return out
    elif method == 'sample':
        # sample from the probabilities
        assert torch.all(vocab_probs >= 0), 'vocab probabilities are not all greater or equal to zero!'
        multi = torch.distributions.Multinomial(1, vocab_probs)
        samp = multi.sample()

        out = torch.argmax(samp, dim=-1)
        assert out.shape == vocab_probs.shape[:-1], 'shape of outputted action should match [batch * 4 * max_length]'
        return out
    else:
        raise NotImplementedError


def discount_rewards(args, vocab_probs, action, my_attention_mask, rewards_d, sentences_type, rewards_c):
    '''

    :param args: should have the gamma value and weighting between rewards_d/ rewards_c
    :param vocab_probs: the probability of each of the words of possible outputs
    :param action: the indices of words were selected to be used in
    :param rewards_c: for each word that was changed give a reward for how correct the discriminator was, same size as input_ids
    :param rewards_d: overall rewards based on labels provided, should be same size as labels
    :return: the error for this (mini)batch for the generator
    '''

    gamma = args.reinforce_gamma
    gamma_tensor = torch.tensor([gamma**i for i in range(action.shape[-1])]).to(args.device)

    assert isinstance(vocab_probs, torch.Tensor)
    probs_selected = vocab_probs.gather(dim=3, index=action.unsqueeze(-1)).squeeze()

    # TODO are classifier inputs just rewarded more??
    rewards = ((sentences_type * rewards_c).view(-1, 1, 1) + rewards_d) * my_attention_mask
    rewards = rewards.view(-1, rewards.shape[-1])
    my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])

    rewards_gamma = torch.empty_like(rewards).to(args.device)
    for j in range(rewards_gamma.shape[-1]):
        rewards_gamma[:, j] = torch.sum(gamma_tensor[:len(gamma_tensor)-j].unsqueeze(0) * rewards[:, j:], dim=1)

    # flipped_rewards = torch.flip(rewards_gamma, [2])
    # cumsum_flipped_rewards = torch.cumsum(flipped_rewards, dim=2)
    # cumsum_rewards = torch.flip(cumsum_flipped_rewards, [2])

    rewards_minus_mean = rewards_gamma - (torch.sum(rewards_gamma * my_attention_mask, dim=-1) / torch.sum(my_attention_mask, dim=-1)).unsqueeze(-1)

    out = -1 * rewards_minus_mean * torch.log(probs_selected.view(-1, probs_selected.shape[-1])) * my_attention_mask

    return torch.sum(out) / torch.sum(my_attention_mask)


def Wloss(preds, labels, my_attention_mask, use_tanh=False):

    if use_tanh:
        tanh = nn.Tanh()
        preds = tanh(preds)

    return -1 * torch.sum(preds * labels * my_attention_mask) / float(my_attention_mask.nonzero().shape[0])


def my_CrossEntropyLoss(scores, labels, neg_one=False):
    CrossEntropy = nn.CrossEntropyLoss(reduction='none')
    multiplier = 1. if neg_one else -1.  # cross entropy already multiplied by neg 1
    out = multiplier * CrossEntropy(scores, labels)
    return out


def my_BinaryCrossEntropyLoss(scores, labels, neg_one=False):
    BCE = nn.BCEWithLogitsLoss(reduction='none')
    multiplier = 1. if neg_one else -1.  # cross entropy already multiplied by neg 1
    out = multiplier * BCE(scores, labels)
    return out



