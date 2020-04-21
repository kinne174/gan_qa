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


class AttentionEssential(nn.Module):
    def __init__(self, config, device):
        super(AttentionEssential, self).__init__()

        self.mu_p = config.mu_p
        self.mask_id = config.mask_id

        self.device = device

    @classmethod
    def from_pretrained(cls, config, device):
        return cls(config, device)

    def forward(self, **kwargs):
        all_attention_mask = kwargs['my_attention_mask']
        out_attention_mask = torch.empty((all_attention_mask.shape[0], all_attention_mask.shape[1], all_attention_mask.shape[2]//2))
        all_tokenizer_attention_mask = kwargs['attention_mask']

        all_input_ids = kwargs['input_ids']
        out_input_ids = torch.empty(all_input_ids.shape)

        for k in range(out_attention_mask.shape[0]):
            for j in range(out_attention_mask.shape[1]):
                attention_mask = all_attention_mask[k, j, :all_attention_mask.shape[2]//2].squeeze()
                shared_tokens = all_attention_mask[k, j, all_attention_mask.shape[2]//2:].squeeze().long()
                tokenizer_attention_mask = all_tokenizer_attention_mask[k, j, :].squeeze()

                input_ids = all_input_ids[k, j, :]

                non_zero_indices = attention_mask.nonzero().reshape((-1))

                num_to_mask = int(torch.sum(tokenizer_attention_mask)*np.random.normal(loc=self.mu_p, scale=min(0.05, self.mu_p/4), size=None))

                non_zeros = attention_mask[non_zero_indices]
                prob_vector = non_zeros/torch.sum(non_zeros)

                non_zero_indices_np = non_zero_indices.cpu().numpy()
                prob_vector_np = prob_vector.cpu().numpy()

                try:

                    weighted_perm = np.random.choice(non_zero_indices_np, size=(non_zero_indices_np.shape[0],), replace=False, p=prob_vector_np)

                except ValueError:

                    non_zero_and_probs = list(map(tuple, zip(non_zero_indices_np, prob_vector_np)))
                    non_zero_and_probs.sort(key=lambda t: t[1])
                    weighted_perm, _ = map(list, zip(*non_zero_and_probs))


                indices_to_mask = weighted_perm[:num_to_mask]
                # shared_tokens_to_mask = [shared_tokens[itm].item() for itm in indices_to_mask]
                # indices_to_mask = [i for i in range(shared_tokens.shape[0]) if shared_tokens[i] in shared_tokens_to_mask]

                new_input_ids = torch.tensor([self.mask_id if i in indices_to_mask else id for i, id in enumerate(input_ids)], dtype=torch.long).reshape((-1))

                new_attention_mask = torch.tensor([1 if i in indices_to_mask else 0 for i in range(attention_mask.shape[0])], dtype=torch.long).reshape((-1,))
                # assert sum(new_attention_mask) >= num_to_mask, 'Sum ({}) is not greater/ equal to num_to_mask ({})'.format(sum(new_attention_mask), num_to_mask)

                out_input_ids[k, j, :] = new_input_ids
                out_attention_mask[k, j, :] = new_attention_mask

        out_input_ids = out_input_ids.long()
        out_attention_mask = out_attention_mask.long()

        out = {}
        for k, v in kwargs.items():
            out[k] = v

        out['my_attention_mask'] = out_attention_mask
        out['input_ids'] = out_input_ids

        assert all([ki in list(out.keys()) for ki in list(kwargs.keys())])
        assert all([ko in list(kwargs.keys()) for ko in list(out.keys())])

        out['discriminator_labels'] = -1 * out_attention_mask

        return out

    def save_pretrained(self):
        raise NotImplementedError


class AttentionEssentialReinforce(nn.Module):
    def __init__(self, config):
        super(AttentionEssentialReinforce, self).__init__()

        self.mu_p = config.mu_p
        self.mask_id = config.mask_id

    @classmethod
    def from_pretrained(cls, config):
        return cls(config)

    def forward(self, input_ids, my_attention_mask):
        # all_attention_mask = kwargs['my_attention_mask']
        out_my_attention_mask = my_attention_mask[:, :, :my_attention_mask.shape[-1] // 2]
        # all_tokenizer_attention_mask = kwargs['attention_mask']

        # all_input_ids = kwargs['input_ids']

        for k in range(out_my_attention_mask.shape[0]):
            for j in range(out_my_attention_mask.shape[1]):
                current_my_attention_mask = my_attention_mask[k, j, :my_attention_mask.shape[-1] // 2].squeeze()
                shared_tokens = my_attention_mask[k, j, my_attention_mask.shape[-1] // 2:].squeeze().long()

                # current_input_ids = input_ids[k, j, :]

                non_zero_indices = current_my_attention_mask.nonzero().reshape((-1))

                num_to_mask = int(self.mu_p*len(non_zero_indices))

                non_zeros = current_my_attention_mask[non_zero_indices]
                prob_vector = non_zeros/torch.sum(non_zeros)

                non_zero_indices_np = non_zero_indices.cpu().numpy()
                prob_vector_np = prob_vector.cpu().numpy()

                try:

                    weighted_perm = np.random.choice(non_zero_indices_np, size=(non_zero_indices_np.shape[0],), replace=False, p=prob_vector_np)

                except ValueError:

                    weighted_perm = np.argpartition(prob_vector_np, num_to_mask)[::-1]

                    # non_zero_and_probs = list(map(tuple, zip(non_zero_indices_np, prob_vector_np)))
                    # non_zero_and_probs.sort(key=lambda t: t[1])
                    # weighted_perm, _ = map(list, zip(*non_zero_and_probs))


                indices_to_mask = weighted_perm[:num_to_mask]
                # shared_tokens_to_mask = [shared_tokens[itm].item() for itm in indices_to_mask]
                # indices_to_mask = [i for i in range(shared_tokens.shape[0]) if shared_tokens[i] in shared_tokens_to_mask]

                for i, id in enumerate(input_ids[k, j, :]):
                    if i in indices_to_mask:
                        input_ids[k, j, i] = self.mask_id

                for i in range(current_my_attention_mask.shape[0]):
                    out_my_attention_mask[k, j, i] = 1 if i in indices_to_mask else 0

        return input_ids, out_my_attention_mask, -1 * out_my_attention_mask

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
    'essential': (AttentionConfig, AttentionEssential),
    'essential-reinforce': (AttentionConfig, AttentionEssentialReinforce)
}