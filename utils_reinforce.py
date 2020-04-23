import torch
import torch.nn as nn
import logging
import math
import numpy as np

# logger
logger = logging.getLogger(__name__)

# sigmoid, softmax
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)


def train_generator(args, input_ids, attention_mask, my_attention_mask, token_type_ids, classification_labels,
                    sentences_type, generatorM, classifierM, discriminatorM, update_step, baseline):

    generatorM.train()
    classifierM.eval()
    discriminatorM.eval()

    # input_ids = fake_inputs['input_ids']
    # attention_mask = fake_inputs['attention_mask']
    # my_attention_mask = fake_inputs['my_attention_mask']
    # token_type_ids = fake_inputs['token_type_ids']
    # classification_labels = fake_inputs['classification_labels']
    # sentences_type = fake_inputs['sentences_type']

    # draw fake probability for words

    fake_vocabulary_probs = generatorM(input_ids, attention_mask, token_type_ids, classification_labels)
    # fake_vocabulary_probs = fake_vocabulary_probs.view(-1, fake_vocabulary_probs.shape[2], fake_vocabulary_probs.shape[3])

    # best_rewards_classifier = 1e10 * torch.ones((1, input_ids.shape[0]))
    # best_index = [None] * input_ids.shape[0]
    # all_loss = []
    # all_fake_input_ids = []
    # all_logits_discriminator = []
    # all_logits_classifier = []
    # for i in range(1):
    # choose words at random or argmax or based on probability
    # update_step = 0 if update_step < 5 else update_step
    # if torch.rand(1).item() > min(0.95, (update_step/args.optimization_steps)**2) or True:
    fake_action = word_index_selector(fake_vocabulary_probs, 'sample')
    # else:
    #     fake_action = word_index_selector(fake_vocabulary_probs, 'argmax')
    fake_input_ids = (my_attention_mask*fake_action + (1 - my_attention_mask)*input_ids).long()
    # throw fake_action to device
    fake_action = fake_action.to(args.device)

    with torch.no_grad():
        # score words with discriminator and classifier
        logits_discriminator = discriminatorM(fake_input_ids)
        logits_classifier = classifierM(fake_input_ids, attention_mask, token_type_ids)

    m_ = (args.rewards_start - args.rewards_stop) / (args.optimization_steps * (2*args.rewards_squeeze - 1))
    b_ = (args.rewards_squeeze * (args.rewards_stop - args.rewards_start) - args.rewards_start) / (2 * args.rewards_squeeze - 1)
    rewards_decay = max(args.rewards_start, min(args.rewards_stop, m_*update_step + b_))
    # rewards_discriminator = my_BinaryCrossEntropyLoss(sigmoid(logits_discriminator), 1-my_attention_mask, neg_one=True)
    rewards_discriminator = (2 - rewards_decay) * 2 * sigmoid(logits_discriminator) - 1
    rewards_classifier = rewards_decay * 2 * softmax(logits_classifier)[classification_labels.nonzero(as_tuple=True)] - 1
    # rewards_classifier = my_CrossEntropyLoss(logits_classifier, classification_labels.nonzero()[:, 1].long(), neg_one=True)
    # False because the generator should be rewarded for fooling the classifier into selecting the wrong options
    #  multiplied by -1 because this is a reward and not an error
    # Same reasons as above for being True # TODO make some annealing on the rewards, probably make it a squished linear function so starts at zero and tops out both in the middle of updating

    # calculate errors
    # have to make sure to use fake_vocabulary_probs to get connection to generator, see eq 9 in scratch gan
    loss = discount_rewards(args, baseline, fake_vocabulary_probs, fake_action, my_attention_mask,
                            rewards_discriminator, sentences_type, rewards_classifier)

    return loss, fake_input_ids, (logits_discriminator, logits_classifier)

        # if torch.any(rewards_classifier.detach().cpu() < best_rewards_classifier):
        #     which_better = (rewards_classifier.detach().cpu() < best_rewards_classifier).nonzero().squeeze().tolist()
        #     for wb in which_better:
        #         best_index[wb] = i
        #         best_rewards_classifier[0, wb] = rewards_classifier.detach().cpu()[wb]

    #     all_loss.append(loss)
    #     all_fake_input_ids.append(fake_input_ids)
    #     all_logits_discriminator.append(logits_discriminator)
    #     all_logits_classifier.append(logits_classifier)
    #
    # return all_loss[best_index[0]].sum(), all_fake_input_ids[best_index[0]], \
    #        (all_logits_discriminator[best_index[0]], all_logits_classifier[best_index[0]])


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

        logits_fake_c = classifierM(fake_input_ids, attention_mask, token_type_ids)
        error_fake_c = my_CrossEntropyLoss(logits_fake_c, classification_labels.nonzero()[:, 1].long(), neg_one=False)
        # True because the classifier should be rewarded for seeing through a poor generator attempt at adversarially
        #  attacking the classifier,
        # False because classifier should not be rewarded for saying something from the generator is correct, otherwise
        #  they work together
        error_fake_c = torch.sum(sentences_type * error_fake_c) / torch.sum(sentences_type)
    else:
        logits_real_c, error_real_c = None, None
        logits_fake_c, error_fake_c = None, None

    logits_real_d = discriminatorM(real_input_ids)
    error_real_d = my_BinaryCrossEntropyLoss(sigmoid(logits_real_d), my_attention_mask, neg_one=True) # training to denote 1 as real
    error_real_d = torch.sum(error_real_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    logits_fake_d = discriminatorM(fake_input_ids)
    # error_fake_d = my_BinaryCrossEntropyLoss(sigmoid(logits_fake_d), my_attention_mask, neg_one=False) # training to denote 0 as fake
    error_fake_d = my_BinaryCrossEntropyLoss(sigmoid(logits_fake_d), my_attention_mask, neg_one=False)
    error_fake_d = torch.sum(error_fake_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    return {'logits_real_c': logits_real_c, 'logits_fake_c': logits_fake_c, 'logits_real_d': logits_real_d, 'logits_fake_d': logits_fake_d}, \
           {'error_real_c': error_real_c, 'error_fake_c': error_fake_c, 'error_real_d': error_real_d, 'error_fake_d': error_fake_d}


def word_index_selector(vocab_probs, method):
    # TODO should I be implenting an annealing completely random choice here?
    if method == 'argmax':
        # do argmax over the third dimension of the probabilities
        out = torch.argmax(vocab_probs, dim=-1)
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


def discount_rewards(args, baseline, vocab_probs, action, my_attention_mask, rewards_d, sentences_type, rewards_c):
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

    rewards = rewards_d * my_attention_mask
    rewards = rewards.view(-1, rewards.shape[-1])
    my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])

    rewards_gamma = torch.empty_like(rewards).to(args.device)
    for j in range(rewards_gamma.shape[-1]):
        rewards_gamma[:, j] = torch.sum(gamma_tensor[:len(gamma_tensor)-j].unsqueeze(0) * rewards[:, j:], dim=1)

    rewards_gamma += ((sentences_type * rewards_c).unsqueeze(1) * torch.ones(*rewards_d.shape[:2]).to(rewards_d.device)).view(-1, 1)

    baseline['num_steps'] += rewards.shape[0]
    baseline['total_mean_rewards'] += torch.sum(torch.sum(rewards_gamma * my_attention_mask, dim=-1) / torch.sum(my_attention_mask, dim=-1))

    if not baseline['num_steps'] == 0:
        new_baseline = baseline['prev'] * args.lambda_baseline + (1 - args.lambda_baseline) * (baseline['total_mean_rewards'] / baseline['num_steps'])
    else:
        new_baseline = 0

    rewards_minus_baseline = rewards_gamma - new_baseline

    loss = torch.mean(torch.sum(-1 * rewards_minus_baseline * torch.log(probs_selected.view(-1, probs_selected.shape[-1])) * my_attention_mask, dim=-1))

    baseline['prev'] = new_baseline

    return loss


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



