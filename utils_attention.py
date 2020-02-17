import numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from nltk.collocations import *
import string
from collections import Counter
import logging

from transformers import PreTrainedTokenizer, PretrainedConfig

logger = logging.getLogger(__name__)

# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_device()


class AttentionNet(torch.nn.Module):

    def __init__(self, config):
        super(AttentionNet, self).__init__()

    @classmethod
    def from_pretrained(cls, config):
        return cls(config)

    def forward(self, input_ids, my_attention_mask, **kwargs):
        # TODOfixed create a model to update my_attention_mask (only where it equals 1) and return all the rest unchanged
        # TODOfixed fix this so it can handle a batch size
        assert input_ids.shape == my_attention_mask.shape

        my_attention_mask = my_attention_mask.view((-1, my_attention_mask.shape[-1]))

        one_indices = [[j for j in range(my_attention_mask.shape[1]) if my_attention_mask[i, j] == 1] for i in range(my_attention_mask.shape[0])]
        chosen_indices = [np.random.choice(one_indices[i], 3).tolist() for i in range(len(one_indices))]
        my_attention_mask = torch.tensor([[1 if j in chosen_indices[i] else 0 for j in range(my_attention_mask.shape[1])] for i in range(my_attention_mask.shape[0])], dtype=torch.long)
        my_attention_mask = my_attention_mask.view(*input_ids.shape)

        assert input_ids.shape == my_attention_mask.shape

        out_dict = {k:v for k,v in kwargs.items() if not k in ['my_attention_mask', 'input_ids']}
        out_dict['my_attention_mask'] = my_attention_mask
        out_dict['input_ids'] = input_ids

        return out_dict

# TODOfixed create attention based on PMI, x from question words: y from answer words, then find them in the context to compute PMI, individual word importance come from matrix of question vs answer words and average/median down the rows and columns
class AttentionPMI(nn.Module):
    # This idea comes from "Combining Retrieval, Statistics, and Inference to Answer Elementary Science Questions" by Clark et al. 2016
    def __init__(self, config):
        super(AttentionPMI, self).__init__()

        self.tokenizer = config.tokenizer
        assert isinstance(self.tokenizer, PreTrainedTokenizer)

        self.max_attention_words = config.max_attention_words

        stop_words = ' '.join(list(stopwords.words('english')))
        stop_words_ids = self.tokenizer.encode(stop_words, add_special_tokens=False)
        special_words_ids = [self.tokenizer.cls_token_id, self.tokenizer.unk_token_id, self.tokenizer.sep_token_id,
                             self.tokenizer.pad_token_id, self.tokenizer.mask_token_id]
        punctuation = ' '.join(list(string.punctuation))
        punctuation_ids = self.tokenizer.encode(punctuation, add_special_tokens=False)

        self.bad_ids = set(stop_words_ids + special_words_ids + punctuation_ids)

        self.window_size = config.window_size
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()

    @classmethod
    def from_pretrained(cls, config):
        return cls(config)

    def forward(self, input_ids, my_attention_mask, **kwargs):

        out_my_attention_mask = torch.LongTensor(*my_attention_mask.shape).to(device)
        for i in range(input_ids.shape[0]):
            # creates a tensor of dimension [4, max length]
            current_input_ids = input_ids[i, :, :].squeeze()

            # retrieve index that the question ends, should be able to use :question_cutoff_ind to retrieve full question
            for k in range(current_input_ids.shape[1]):
                if current_input_ids[0, k] == current_input_ids[1, k] == current_input_ids[2, k] == current_input_ids[3, k]:
                    continue
                question_cutoff_ind = k-1
                break

            # retrieve unique question ids
            question_ids = list(set(current_input_ids[1, :question_cutoff_ind].tolist()))

            # using default form of my_attention_mask which will have 0s for QA part and 1s for context find the first index that has a 1
            # and the answer will be between this and the question cutoff index, get unique ids
            answer_cutoff_inds = [min(my_attention_mask[i, j, :].nonzero()) for j in range(my_attention_mask.shape[1])]
            answer_ids = [list(set(current_input_ids[j, question_cutoff_ind:answer_cutoff_inds[j]].tolist())) for j in range(current_input_ids.shape[0])]

            # find context using similar logic as before, max will either be when the padding starts or when the context ends
            context_cutoff_inds = [max(my_attention_mask[i, j, :].nonzero()) for j in range(my_attention_mask.shape[1])]
            context_ids = [current_input_ids[j, answer_cutoff_inds[j]:context_cutoff_inds[j]].tolist() for j in range(current_input_ids.shape[0])]

            # delete stop words, punctuation and special tokenizer tokens
            question_ids = [qi for qi in question_ids if qi not in self.bad_ids]
            answer_ids_temp = [[ai for ai in sub_answer_ids if ai not in self.bad_ids] for sub_answer_ids in answer_ids]
            answer_ids = [ai_t if len(ai_t) != 0 else ai for ai_t, ai in zip(answer_ids_temp, answer_ids)]

            # save index of ids in tensor for creating the outputted my_attention_mask, use a tuple of (index, id), needed for context only
            context_ids_tups = [[(ci, ind + answer_cutoff_inds[j] + 1) for ind, ci in enumerate(sub_context_ids)] for j, sub_context_ids in enumerate(context_ids)]

            # TODO should I be deleting stop words here? If I'm using a word window in the next step?
            # context_ids = [[ci for ci in sub_context_ids if ci not in self.bad_ids] for sub_context_ids in context_ids]

            # temporary tensor of dimension [4, max length] to hold output masks for all options in current question
            sub_out_my_attention_mask = torch.LongTensor(*current_input_ids.shape).to(device)

            # do a double for loop over each word-pair in question and answer to find words with the largest PMI
            for ii, (sub_answer_ids, sub_context_ids, sub_context_ids_tups) in enumerate(zip(answer_ids, context_ids, context_ids_tups)):

                # initialize different matrices for answer and question just in case a word is not seen in the context it won't affect its bigram pair's score
                PMI_matrix_a = torch.zeros((len(sub_answer_ids), len(question_ids))).to(device)
                PMI_matrix_q = torch.zeros((len(question_ids), len(sub_answer_ids))).to(device)

                # for the denominator initialize a counter of the context words
                context_counter = Counter(sub_context_ids)

                # should be able to use this to find the PMI of bigrams
                finder = BigramCollocationFinder.from_words(sub_context_ids, window_size=self.window_size)

                # tempororary tensor of dimension [max length] to hold individual attention masks of question/answer/contexts
                sub_my_attention = torch.LongTensor(current_input_ids.shape[-1],).to(device)

                # TODOfixed if none of the answer words are in the context then PMI_matrix_q should be a count of the words in the context, similar for question words
                if all([ai not in context_counter for ai in sub_answer_ids]) and all([qi not in context_counter for qi in question_ids]):
                    PMI_matrix_a = torch.zeros((max(len(sub_answer_ids), 1), 1))  # to account for no answer ids in context
                    PMI_matrix_q = torch.zeros((max(len(question_ids), 1), 1))

                elif all([ai not in context_counter for ai in sub_answer_ids]):
                    PMI_matrix_a = torch.zeros((max(len(sub_answer_ids), 1), 1))  # to account for no answer ids in context
                    PMI_matrix_q = torch.tensor([context_counter[qi] for qi in question_ids], dtype=torch.float).unsqueeze(1)

                elif all([qi not in context_counter for qi in question_ids]):
                    PMI_matrix_q = torch.zeros((max(len(question_ids), 1), 1))
                    PMI_matrix_a = torch.tensor([context_counter[ai] for ai in sub_answer_ids], dtype=torch.float).unsqueeze(1)

                else:
                    for a_ind, ai in enumerate(sub_answer_ids):
                        for q_ind, qi in enumerate(question_ids):
                            if ai == qi:
                                # this shouldn't add or subtract from the score of this word
                                continue

                            # don't pick one of these words as it doesn't appear in the context
                            if ai not in context_counter:
                                PMI_matrix_a[a_ind, q_ind] = 0
                            if qi not in context_counter:
                                PMI_matrix_q[q_ind, a_ind] = 0
                                continue

                            # find bi gram score within a window
                            p_ai_qi = finder.score_ngram(self.bigram_measures.pmi, ai, qi)
                            p_ai_qi = .1 if p_ai_qi is None else 2**p_ai_qi  # 2 because nltk uses log2

                            # assign PMI to each matrix
                            PMI_matrix_a[a_ind, q_ind] = p_ai_qi
                            PMI_matrix_q[q_ind, a_ind] = p_ai_qi

                # find max over the columns for each matrix
                # be tensors of dimension [len(sub_answer_ids)] and [len(question_ids)] respectively
                PMI_max_a, _ = torch.max(PMI_matrix_a, dim=1)
                PMI_max_q, _ = torch.max(PMI_matrix_q, dim=1)

                # combine and find indices of largest PMI score from left to right
                # both should be tensors of dimension [len(sub_answer_ids) + len(question_ids)]
                PMI_max = torch.cat((PMI_max_q, PMI_max_a), dim=0)
                PMI_indices = torch.argsort(PMI_max, descending=True)

                # concatenating list of question and answer ids similar to above
                question_answer_ids = question_ids + sub_answer_ids
                if not question_answer_ids:
                    question_answer_ids = [0, 0] # only if no ids in context from BOTH question and answer ids

                # keep track of how many words will be masked in context and the words selected to be masked
                masked_context_ids = []

                for max_ind in PMI_indices[:self.max_attention_words]:
                    # cycle through max words for number of unique words to mask
                    try:
                        max_word = question_answer_ids[max_ind.item()]
                    except IndexError: # TODO this is in case only one of question Ids or answer Ids are not in the context, quick fix SHOULD NOT BE LEFT
                        continue
                    masked_context_ids.append(max_word)

                # if the number of unique words from the answer and questions is less than the max_attention_words
                # provided by the user then add in the appropriate amount randomly from the context
                num_words_found_in_context = PMI_max.nonzero().shape[0]
                if num_words_found_in_context < self.max_attention_words:
                    masked_context_ids.extend(np.random.choice(list(context_counter.keys()), self.max_attention_words - num_words_found_in_context, replace=False).tolist())

                # find which indices to mask based on the tuples of context words and context indices
                # indices should be in relation to the [0,.., max length]
                # no set size for this tensor
                indices_to_mask = torch.tensor([c_tup[1] for c_tup in sub_context_ids_tups if c_tup[0] in masked_context_ids], dtype=torch.long).to(device)  # might need to make this 2D

                if torch.sum(indices_to_mask) == 0:
                    print('hi')

                # place ones in the positions to mask
                sub_my_attention.zero_()
                sub_my_attention.scatter_(0, indices_to_mask, 1)

                # save single attention mask in temporary tensor
                sub_out_my_attention_mask[ii, :] = sub_my_attention

            # update output tensor with whole question attention mask
            assert out_my_attention_mask.shape[1:] == sub_out_my_attention_mask.shape
            out_my_attention_mask[i, :, :] = sub_out_my_attention_mask

        assert out_my_attention_mask.shape == input_ids.shape
        assert out_my_attention_mask.dtype == torch.long

        out_dict = {k: v for k, v in kwargs.items()}

        out_dict['input_ids'] = input_ids.detach()
        out_dict['my_attention_mask'] = out_my_attention_mask.detach()

        return out_dict

class AttentionConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super(AttentionConfig, self).__init__()

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls(**kwargs)


attention_models_and_config_classes = {
    'random': (AttentionConfig, AttentionNet),
    'PMI': (AttentionConfig, AttentionPMI),
}