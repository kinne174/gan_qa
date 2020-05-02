import torch
import torch.nn as nn
import logging
import math

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

    batch_size = input_ids.shape[0]

    # draw fake probability for words

    fake_vocabulary_logprobs = generatorM(input_ids, attention_mask, token_type_ids, classification_labels)
    # fake_vocabulary_probs = fake_vocabulary_probs.view(-1, fake_vocabulary_probs.shape[2], fake_vocabulary_probs.shape[3])

    all_log_probs = []
    all_fake_input_ids = []
    all_fake_action = []
    for _ in range(args.generator_resampling_steps):
        # if torch.rand(1).item() > min(0.95, (update_step/args.optimization_steps)**2) or True:
        fake_action, log_probs = word_index_selector(fake_vocabulary_logprobs, 'sample_log', update_step)
        # else:
        #     fake_action = word_index_selector(fake_vocabulary_probs, 'argmax')
        fake_input_ids = (my_attention_mask*fake_action + (1 - my_attention_mask)*input_ids).long()
        # throw fake_action to device
        fake_action = fake_action.to(args.device)

        all_fake_input_ids.append(fake_input_ids)
        all_log_probs.append(log_probs)
        all_fake_action.append(fake_action)

    log_probs = torch.cat(all_log_probs, dim=0)
    fake_input_ids = torch.cat(all_fake_input_ids, dim=0)
    fake_action = torch.cat(all_fake_action, dim=0)
    attention_mask = torch.cat([attention_mask]*args.generator_resampling_steps, dim=0)
    token_type_ids = torch.cat([token_type_ids]*args.generator_resampling_steps, dim=0)
    classification_labels = torch.cat([classification_labels]*args.generator_resampling_steps, dim=0)

    with torch.no_grad():
        # score words with discriminator and classifier
        logits_discriminator = discriminatorM(fake_input_ids)
        logits_classifier = classifierM(fake_input_ids, attention_mask, token_type_ids)

    m_ = (args.rewards_start - args.rewards_stop) / (args.optimization_steps * (2*args.rewards_squeeze - 1))
    b_ = (args.rewards_squeeze * (args.rewards_stop + args.rewards_start) - args.rewards_start) / (2 * args.rewards_squeeze - 1)
    rewards_decay = max(args.rewards_start, min(args.rewards_stop, m_*update_step + b_))
    # rewards_discriminator = my_BinaryCrossEntropyLoss(sigmoid(logits_discriminator), 1-my_attention_mask, neg_one=True)
    rewards_discriminator = 2. * sigmoid(logits_discriminator) - 1.
    rd_mean = torch.mean(rewards_discriminator)
    rd_std = torch.std(rewards_discriminator)

    rewards_classifier = 2. * softmax(logits_classifier)[classification_labels.nonzero(as_tuple=True)] - 1.
    rc_mean = torch.mean(rewards_classifier)
    rc_std = torch.std(rewards_classifier)

    rewards_classifier = ((rewards_classifier - rc_mean) / rc_std) * rd_std + rd_mean

    rewards_discriminator *= (2 - rewards_decay)
    rewards_classifier *= rewards_decay
    # rewards_classifier = my_CrossEntropyLoss(logits_classifier, classification_labels.nonzero()[:, 1].long(), neg_one=True)
    # False because the generator should be rewarded for fooling the classifier into selecting the wrong options
    #  multiplied by -1 because this is a reward and not an error
    # Same reasons as above for being True

    my_attention_mask = torch.cat([my_attention_mask]*args.generator_resampling_steps, dim=0)
    sentences_type = torch.cat([sentences_type]*args.generator_resampling_steps, dim=0)

    # calculate errors
    # have to make sure to use fake_vocabulary_probs to get connection to generator, see eq 9 in scratch gan
    loss = discount_rewards(args, baseline, log_probs, fake_action, my_attention_mask,
                            rewards_discriminator, sentences_type, rewards_classifier)

    random_ind = torch.LongTensor(1).random_(args.generator_resampling_steps).item() * batch_size
    a = torch.arange(random_ind, random_ind+batch_size)
    fake_input_ids = fake_input_ids[a, ...]
    logits_discriminator = logits_discriminator[a, ...]
    logits_classifier = logits_classifier[a, ...]

    return loss, fake_input_ids, (logits_discriminator, logits_classifier)


def train_classifier_discriminator(args, fake_input_ids, real_input_ids, attention_mask, my_attention_mask,
                                   token_type_ids, classification_labels, sentences_type, discriminatorM, classifierM):

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
    error_fake_d = my_BinaryCrossEntropyLoss(sigmoid(logits_fake_d), my_attention_mask, neg_one=False) # training to denote 0 as fake
    error_fake_d = torch.sum(error_fake_d[my_attention_mask.nonzero(as_tuple=True)]) / float(my_attention_mask.nonzero().shape[0])

    return {'logits_real_c': logits_real_c, 'logits_fake_c': logits_fake_c, 'logits_real_d': logits_real_d, 'logits_fake_d': logits_fake_d}, \
           {'error_real_c': error_real_c, 'error_fake_c': error_fake_c, 'error_real_d': error_real_d, 'error_fake_d': error_fake_d}


def word_index_selector(vocab_probs, method, update_step):
    if method == 'argmax':
        # do argmax over the third dimension of the probabilities
        out = torch.argmax(vocab_probs, dim=-1)
        assert out.shape == vocab_probs.shape[:-1], 'shape of outputted action should match [batch * 4 * max_length]'
        return out, None
    elif method == 'sample':
        # sample from the probabilities
        vocab_probs = torch.exp(vocab_probs)
        assert torch.all(vocab_probs >= 0), 'vocab probabilities are not all greater or equal to zero!'
        multi = torch.distributions.Multinomial(1, vocab_probs)
        samp = multi.sample()

        out = torch.argmax(samp, dim=-1)
        assert out.shape == vocab_probs.shape[:-1], 'shape of outputted action should match [batch * 4 * max_length]'
        probs_selected = torch.log(vocab_probs.gather(dim=3, index=out.unsqueeze(-1)).squeeze())
        return out, probs_selected
    elif method == 'sample_log':
        cat = torch.distributions.Categorical(logits=vocab_probs)
        samp = cat.sample()  # TODO look at soft actor critic methods

        # values, _ = torch.topk(vocab_probs, k=2)
        # inidices_to_completely_randomize = torch.where((torch.exp(values[..., 0].squeeze()) - torch.exp(values[..., 1].squeeze())) > .999,
        #                                                torch.ones_like(samp), torch.zeros_like(samp))
        #
        # if (inidices_to_completely_randomize.nonzero().shape[0] / samp.numel()) > 0.99 and update_step > -1:
        #     cat_random = torch.distributions.Categorical(probs=torch.ones_like(vocab_probs))
        #     samp_random = cat_random.sample()
        #     logger.warning('Using completely randomized words!')
        #
        #     # start = 0.01
        #     # stop = 0.005
        #     # a = (args.optimization_steps * math.log(start) - math.log(stop)) / math.log(stop / start)
        #     # t = -1 * math.log(start) / (a + 1)
        #     # limit = math.exp(-1 * t * (a + update_step + 1))
        #
        #     # indices_to_replace = inidices_to_completely_randomize * (torch.rand_like(inidices_to_completely_randomize, dtype=torch.float) < limit)
        #
        #     samp = samp * (1 - inidices_to_completely_randomize) + samp_random * inidices_to_completely_randomize # TODO need to think about this again, allow more exploration?

        log_probs = cat.log_prob(samp)

        return samp.long(), log_probs
    else:
        raise NotImplementedError


def discount_rewards(args, baseline, log_vocab_probs, action, my_attention_mask, rewards_d, sentences_type, rewards_c):
    '''

    :param args: should have the gamma value and weighting between rewards_d/ rewards_c
    :param vocab_probs: the probability of each of the words of possible outputs
    :param action: the indices of words were selected to be used in
    :param rewards_c: for each word that was changed give a reward for how correct the discriminator was, same size as input_ids
    :param rewards_d: overall rewards based on labels provided, should be same size as labels
    :return: the error for this (mini)batch for the generator
    '''

    gamma = args.reinforce_gamma
    gamma_tensor = gamma ** torch.arange(action.shape[-1], dtype=torch.float).to(args.device)

    # old_rewards = rewards_d * my_attention_mask
    # old_rewards = old_rewards.view(-1, old_rewards.shape[-1])
    rewards = (rewards_d + rewards_c.view(-1, 1, 1)) * my_attention_mask
    rewards = rewards.view(-1, rewards.shape[-1])
    my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])

    attention_mask_rewards = rewards[my_attention_mask.nonzero(as_tuple=True)]

    rewards_gamma = torch.empty_like(rewards).to(args.device)
    for j in range(rewards_gamma.shape[-1]):
        if j is not 0:
            gamma_tensor = torch.cat((gamma_tensor[0].unsqueeze(0) * gamma, gamma_tensor[:-1]))
        rewards_gamma[:, j] = torch.sum(gamma_tensor.unsqueeze(0) * rewards, dim=1)

    attention_mask_gamma = rewards_gamma[my_attention_mask.nonzero(as_tuple=True)]

    # old_rewards_gamma = torch.empty_like(rewards).to(args.device)
    # for j in range(old_rewards_gamma.shape[-1]):
    #     old_rewards_gamma[:, j] = torch.sum(gamma_tensor[:len(gamma_tensor)-j].unsqueeze(0) * old_rewards[:, j:], dim=1)

    # old_rewards_gamma += ((sentences_type * rewards_c).unsqueeze(1) * torch.ones(*rewards_d.shape[:2]).to(rewards_d.device)).view(-1, 1)
    # mean_rewards_gamma = torch.sum(rewards_gamma * my_attention_mask, dim=-1) / (torch.sum(my_attention_mask, dim=-1) + 1e-6)
    # mean_old_rewards_gamma = torch.sum(old_rewards_gamma * my_attention_mask, dim=-1) / (torch.sum(my_attention_mask, dim=-1) + 1e-6)

    baseline['num_steps'] += rewards.shape[0]
    baseline['total_mean_rewards'] += torch.sum(torch.sum(rewards_gamma * my_attention_mask, dim=-1) / (torch.sum(my_attention_mask, dim=-1) + 1e-6))

    new_baseline = baseline['prev'] * args.lambda_baseline + (1 - args.lambda_baseline) * (baseline['total_mean_rewards'] / baseline['num_steps'])

    rewards_minus_baseline = rewards_gamma - new_baseline

    attention_mask_minus_baseline = rewards_minus_baseline[my_attention_mask.nonzero(as_tuple=True)]
    s = torch.sum(attention_mask_minus_baseline)
    mi = torch.min(attention_mask_minus_baseline)
    ma = torch.max(attention_mask_minus_baseline)
    avg = torch.mean(attention_mask_minus_baseline)

    log_vocab_probs = log_vocab_probs.view(-1, log_vocab_probs.shape[-1]) * my_attention_mask
    loss = torch.mean(torch.sum(-1 * rewards_minus_baseline * log_vocab_probs, dim=-1))

    attention_mask_log_probs = log_vocab_probs.view(-1, log_vocab_probs.shape[-1])[my_attention_mask.nonzero(as_tuple=True)]

    baseline['prev'] = new_baseline

    if torch.any(torch.isnan(baseline['total_mean_rewards'])):
        print('hi')
        return None

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



