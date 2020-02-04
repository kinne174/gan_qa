import numpy as np
from utils_embedding_model import ArcFeature
import torch
import torch.nn as nn


class AttentionNet(torch.nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()

        # self.in_features = config.in_features_attention
        #
        # self.linear1 = nn.Sequential(
        #     nn.Linear(self.in_features, 124),
        #     nn.BatchNorm1d(124),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.out = nn.Sequential(
        #     nn.Linear(124, self.in_features),
        #     nn.BatchNorm1d(self.in_features),
        #     nn.Tanh()
        # )

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
#
# def int_list(l):
#     return [int(o) for o in l]


# def randomize_attention_loader(examples, model=None):
#     if model is None:
#         for examp in examples:
#             assert isinstance(examp, ArcExample)
#             examp.context_attention_masks = [list(np.random.choice((-1, 0, 1), (len(c),))) for c in examp.contexts]
#     else:
#         num_in_features = model.in_features
#         x = torch.randn((len(examples)*4, num_in_features))
#         output = model(x)
#         output = output.view((len(examples), 4, -1))
#
#         for ii, examp in enumerate(examples):
#             examp.context_attention_masks = [int_list(output[ii, jj, :].tolist()) for jj in range(4)]
#
#             assert all([len(cam) == len(con) for cam, con in zip(examp.context_attention_masks, examp.contexts)])
#
#     return examples


def attention_loader(features, context_attention_masks):
    # assume context attention masks is a tensor of size (4*len(features), universal context_length)
    for i, f in enumerate(features):
        assert isinstance(f, ArcFeature)
        cf_list = f.choices_features # list of dictionaries with keys inputids, input_mask, segment_ids, and attention_mask
        for j, cf in enumerate(cf_list):
            # may need to do some clean up here depending on how the output of the attention model looks
            cf['attention_mask'] = list(context_attention_masks[(i*4)+j, :].squeeze().tolist())

    return features


attention_models = {
    'linear': AttentionNet
}