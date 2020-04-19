import torch
import torch.nn as nn
import os
import logging
import numpy as np
import json
import random

logger = logging.getLogger(__name__)

def train_generator(args, fake_inputs, generatorM, classifierM, discriminatorM):

    generatorM.train()

    input_ids = fake_inputs['input_ids']
    attention_mask = fake_inputs['attention_mask']
    my_attention_mask = fake_inputs['my_attention_mask']
    token_type_ids = fake_inputs['token_type_ids']
    classification_labels = fake_inputs['classification_labels']
    sentences_type = fake_inputs['sentences_type']

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

    rewards_classifier = my_CrossEntropyLoss(logits_classifier, neg_one=True)

    # calculate errors
    # have to make sure touse fake_vocabulary_probs to get connection to generator, see eq 9 in scratch gan
    loss = discount_rewards(args, fake_vocabulary_probs, fake_action, logits_discriminator, sentences_type, rewards_classifier)

    return loss, fake_input_ids, (logits_discriminator, logits_classifier)

def train_classifier_discriminator(args, fake_inputs, real_inputs, discriminatorM, classifierM):

    fake_input_ids = fake_inputs['input_ids']
    real_input_ids = real_inputs['input_ids']
    attention_mask = fake_inputs['attention_mask']
    my_attention_mask = fake_inputs['my_attention_mask']
    token_type_ids = fake_inputs['token_type_ids']
    classification_labels = real_inputs['classification_labels']
    sentences_type = fake_inputs['sentences_type']

    if sentences_type.nonzero().shape[0]:
        logits_real_c = classifierM(real_input_ids, attention_mask, token_type_ids)
        error_real_c = my_CrossEntropyLoss(logits_real_c, neg_one=False)
        error_real_c = torch.sum(sentences_type * error_real_c[(classification_labels == 1).nonzero(as_tuple=True)]) / torch.sum(sentences_type)
    else:
        logits_real_c, error_real_c = None, None

    logits_real_d = discriminatorM(real_input_ids)
    # error_real_d = Wloss(logits_real_d, real_discriminator_labels, my_attention_mask, use_tanh=True)
    error_real_d = my_CrossEntropyLoss(logits_real_d, neg_one=True)
    error_real_d = (error_real_d * my_attention_mask) / float(my_attention_mask.nonzero().shape[0])

    logits_fake_d = discriminatorM(fake_input_ids)
    # error_fake_d = Wloss(logits_fake_d, fake_discriminator_labels, my_attention_mask, use_tanh=True)
    error_fake_d = my_CrossEntropyLoss(logits_fake_d, neg_one=False)
    error_fake_d = (error_fake_d * my_attention_mask) / float(my_attention_mask.nonzero().shape[0])

    return (logits_real_c, logits_real_d, logits_fake_d), (error_real_c, error_real_d, error_fake_d)


def word_index_selector(vocab_probs, method):
    # TODO should I be implenting an annealing completely random choice here?
    if method == 'argmax':
        # do argmax over the third dimension of the probabilities
        out = torch.argmax(vocab_probs, -1)
        assert out.shape == vocab_probs.shape[:-1]
        return out
        pass
    elif method == 'sample':
        # sample from the probabilities
        out = torch.empty(*vocab_probs.shape[:-1])
        for i in range(vocab_probs.shape[0]):
            for j in range(vocab_probs.shape[1]):
                for k in range(vocab_probs.shape[2]):
                    out[i, j, k] = int(random.choices(range(vocab_probs.shape[-1]), weights=vocab_probs[i, j, k, :]))
        assert out.shape == vocab_probs.shape[:-1]
        return out
    else:
        raise NotImplementedError

def discount_rewards(args, vocab_probs, action, rewards_d, sentences_type, rewards_c):
    '''

    :param args: should have the gamma value and weighting between rewards_d/ rewards_c
    :param vocab_probs: the probability of each of the words of possible outputs
    :param action: the indices of words were selected to be used in
    :param rewards_c: for each word that was changed give a reward for how correct the discriminator was, same size as input_ids
    :param rewards_d: overall rewards based on labels provided, should be same size as labels
    :return: the error for this (mini)batch for the generator
    '''

    gamma = args.reinforce_gamma
    gamma_tensor = torch.tensor([gamma**i for i in range(vocab_probs.shape[-1])]).view(1, 1, vocab_probs.shape[-1])

    assert isinstance(vocab_probs, torch.Tensor)
    probs_selected = vocab_probs.gather(dim=3, index=action).squeeze()

    # TODO this is different than equation 7 in scratch gan, may want to find a way to do it like them,
    #  could be done with a rotating idea with gammas before cumsum
    # TODO are classifier inputs just rewarded more??
    rewards = (sentences_type * rewards_c).unsqueeze(2) + rewards_d

    rewards_gamma = rewards * gamma_tensor

    flipped_rewards = torch.flip(rewards_gamma, [2])
    cumsum_flipped_rewards = torch.cumsum(flipped_rewards, dim=2)
    cumsum_rewards = torch.flip(cumsum_flipped_rewards, [2])

    rewards_minus = cumsum_rewards - torch.mean(cumsum_rewards, dim=2).unsqueeze(2)

    out = -1 * rewards_minus * torch.log(probs_selected)

    # TODO more thought needs to be put into the rewards, do I use labels for discriminator? do I use equation
    #  6 from scratch gan to compute rewards? do I do it only for generated words or for all words?

    return out.mean()


def Wloss(preds, labels, my_attention_mask, use_tanh=False):

    if use_tanh:
        tanh = nn.Tanh()
        preds = tanh(preds)

    return -1 * torch.sum(preds * labels * my_attention_mask) / float(my_attention_mask.nonzero().shape[0])


def my_CrossEntropyLoss(scores, neg_one=False):
    multiplier = -1 if neg_one else 1
    return multiplier * torch.log(torch.exp(scores) / torch.sum(torch.exp(scores), dim=-1))





