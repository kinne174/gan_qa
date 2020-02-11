import torch
import torch.nn as nn
from transformers import (BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM,
                          BertConfig, DistilBertConfig, RobertaConfig, AlbertConfig,
                          AlbertForMaskedLM)
from transformers import PretrainedConfig
from torch.nn import functional as F
import logging
from tqdm import trange

logger = logging.getLogger(__name__)

# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_device()


# from https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

# from https://discuss.pytorch.org/t/vae-gumbel-softmax/16838
def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0, 1)
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=0.5, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = config.encoder
        self.decoder = config.decoder

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.criterion = nn.CrossEntropyLoss()

    @classmethod
    def from_pretrained(cls, config):
        return cls(config)

    def forward(self, input_ids, my_attention_mask, **kwargs):

        temp_input_ids = torch.t(input_ids.view(-1, input_ids.shape[-1]))  # [max len, batch size]
        temp_my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])  # [batch size, max len]

        batch_size = temp_input_ids.shape[1]  # this should be 4*batch size
        max_len = temp_input_ids.shape[0]  # this should be max length
        out_len = max(torch.sum(temp_my_attention_mask, dim=1)) # number of maximum masked tokens
        vocab_size = self.decoder.output_dim  # this should be the number of possible vocab words

        # tensor to store decoder outputs [max attention masks, batch size, vocab size]
        outputs = torch.rand((out_len, batch_size, vocab_size)).to(device)
        # outputs = torch.rand((max_len, batch_size, vocab_size)).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(temp_input_ids)

        # first input to the decoder is the <sos> tokens
        input = temp_input_ids[0, :]

        # for t in trange(1, max_len):
        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)  # output is [4*batch size, vocab size]

            # place predictions in a tensor holding predictions for each token
            # outputs[t] = output
            for j in range(batch_size):
                if temp_my_attention_mask[j, t] == 0:
                    continue
                else:
                    i = torch.sum(temp_my_attention_mask[j, :t])
                    outputs[i, j, :] = output[j, :]

            # decide if we are going to use teacher forcing or not
            # teacher_force = random.random() < teacher_forcing_ratio

            # get the true token to supply next
            input = temp_input_ids[t, :]

        # should give dimension [max attention masks, 4*batch size, vocab size] with one hot vectors along the third dimension
        gs = gumbel_softmax(outputs, hard=True).to(device)

        # should start with dimension [4*batch_size, max length, vocab size] with one hot vectors along the third dimension
        # one hot vectors are indicative of the word ids to be used by the classifier
        onehots = torch.zeros((batch_size, max_len, vocab_size)).to(device)
        onehot = torch.FloatTensor(1, vocab_size)
        for i in range(batch_size):
            for j in range(max_len):
                if temp_my_attention_mask[i, j] == 1:
                    k = torch.sum(temp_my_attention_mask[i, :j])
                    onehots[i, j, :] = gs[k, i, :]
                else:
                    onehot.zero_()
                    onehot.scatter_(1, torch.tensor([temp_input_ids[j, i]]).unsqueeze(0), 1)
                    onehots[i, j, :] = onehot
        # change to dimension [4*batch size*max length, vocab size] to make multiplying by embeddings in classifier easier
        onehots = onehots.view(-1, onehots.shape[-1])
        # change to sparse for memory savage...?? not sure if that actually helps
        onehots = onehots.to_sparse()

        # # should give dimension [max attention masks, 4*batch size, vocab size] with 0,1,2...,vocab size along the third dimension
        # input_indicators = torch.arange(0, gs.shape[2]).expand_as(gs)
        #
        # # should give dimension [max attention masks, 4*batch size] with prediction of word token
        # summed = torch.sum(gs * input_indicators, dim=2)
        # assert torch.max(summed) <= vocab_size
        #
        # # gives dimension [max length, batch size] with 0s at non masked tokens and new token at masked tokens
        # new_sentences = torch.zeros(*temp_input_ids.shape)
        # for j in range(batch_size):
        #     for k in range(max_len):
        #         if temp_my_attention_mask[j, k] == 0:
        #             continue
        #         else:
        #             i = torch.sum(temp_my_attention_mask[j, :k])
        #             new_sentences[k, j] = summed[i, j]
        # # new_sentences = summed*torch.t(temp_attention_mask)
        #
        # # gives dimension [batch size, max length] with 0s at masked tokens and remaining tokens at non masked tokens
        # remaining_sentences = torch.t(temp_input_ids) * (torch.ones(*temp_my_attention_mask.shape) - temp_my_attention_mask)
        #
        # assert remaining_sentences.shape == torch.t(new_sentences).shape, 'shape of remaining is {} and new is {}'.format(*remaining_sentences.shape, *torch.t(new_sentences).shape)
        #
        # # adding gives new sentences with replaced tokens [batch size, max length]
        # out = remaining_sentences + torch.t(new_sentences)

        out_dict = {k: v for k, v in kwargs.items()}
        # reshape input ids to resemble known form
        # out_dict['input_ids'] = out.view((-1, 4, input_ids.shape[0]))
        out_dict['input_ids'] = input_ids
        out_dict['inputs_embeds'] = onehots

        return out_dict


class GeneratorConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super(GeneratorConfig, self).__init__()

        if kwargs['pretrained_model_name_or_path'] == 'seq':
            self.encoder = Encoder(input_dim=kwargs['input_dim'], emb_dim=5, hid_dim=200, n_layers=1, dropout=0.0)
            self.decoder = Decoder(output_dim=kwargs['input_dim'], emb_dim=5, hid_dim=200, n_layers=1, dropout=0.0)

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_pretrained(cls, **kwargs):
        # if kwargs['pretrained_model_name_or_path'] == 'linear':
        #     self.in_features = 100
        #     self.hidden_features = 200
        # if kwargs['pretrained_model_name_or_path'] == 'seq':
        #     self.encoder = Encoder(input_dim=kwargs['input_dim'], emb_dim=100, hid_dim=200, n_layers=1, dropout=0.0)
        #     self.decoder = Decoder(output_dim=kwargs['input_dim'], emb_dim=100, hid_dim=200, n_layers=1, dropout=0.0)
        #     self.device = kwargs['device']
        # else:
        #     raise Exception('Not implemented yet')
        return cls(**kwargs)


class MyAlbertForMaskedLM(nn.Module):
    def __init__(self, pretrained_model_name_or_path, config):
        super(MyAlbertForMaskedLM, self).__init__()
        self.albert = AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(pretrained_model_name_or_path, config)

    def forward(self, input_ids, my_attention_mask, attention_mask, token_type_ids, **kwargs):

        # change from dimension [batch size, 4, max length] to [4*batch size, max length]
        temp_input_ids = input_ids.view(-1, input_ids.shape[-1])
        temp_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        temp_token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        temp_my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])  # [batch size, max len]

        batch_size = temp_input_ids.shape[0]  # this should be 4*batch size
        max_len = temp_input_ids.shape[1]  # this should be max length
        out_len = max(torch.sum(temp_my_attention_mask, dim=1))  # number of maximum masked tokens

        # outputs dimension [4*batch size, max length, vocab size] of before softmax scores for each word
        albert_outputs = self.albert(input_ids=temp_input_ids.long(),
                                      attention_mask=temp_attention_mask,
                                      token_type_ids=temp_token_type_ids)
        # albert_outputs = [torch.rand((4*batch_size, max_len, 30000))]

        prediction_scores = albert_outputs[0]
        vocab_size = prediction_scores.shape[-1]

        # tensor to store decoder outputs [max attention masks, batch size, vocab size]
        outputs = torch.rand((out_len, batch_size, vocab_size)).to(device)

        for t in range(1, max_len):
            # place predictions in a tensor holding predictions for each token
            for j in range(batch_size):
                if temp_my_attention_mask[j, t] == 0:
                    continue
                else:
                    i = torch.sum(temp_my_attention_mask[j, :t])
                    outputs[i, j, :] = prediction_scores[j, t, :]

        # should give dimension [max attention masks, 4*batch size, vocab size] with one hot vectors along the third dimension
        gs = gumbel_softmax(outputs, hard=True)

        # should start with dimension [4*batch_size, max length, vocab size] with one hot vectors along the third dimension
        # one hot vectors are indicative of the word ids to be used by the classifier
        onehots = torch.zeros((batch_size, max_len, vocab_size))
        onehot = torch.FloatTensor(1, vocab_size)
        for i in range(batch_size):
            for j in range(max_len):
                if temp_my_attention_mask[i, j] == 1:
                    k = torch.sum(temp_my_attention_mask[i, :j])
                    onehots[i, j, :] = gs[k, i, :]
                else:
                    onehot.zero_()
                    onehot.scatter_(1, torch.tensor([temp_input_ids[j, i]]).unsqueeze(0), 1)
                    onehots[i, j, :] = onehot
        # change to dimension [4*batch size*max length, vocab size] to make multiplying by embeddings in classifier easier
        onehots = onehots.view(-1, onehots.shape[-1])
        # change to sparse for memory savage...?? not sure if that actually helps
        onehots = onehots.to_sparse()

        # # should give dimension [max attention masks, 4*batch size, vocab size] with 0,1,2...,vocab size along the third dimension
        # input_indicators = torch.arange(0, gs.shape[2]).expand_as(gs)
        #
        # # should give dimension [max attention masks, 4*batch size] with prediction of word token
        # summed = torch.sum(gs * input_indicators, dim=2)
        # assert torch.max(summed) <= vocab_size
        #
        # # gives dimension [4*batch size, max length] with 0s at non masked tokens and new token at masked tokens
        # new_sentences = torch.zeros(*temp_input_ids.shape)
        # for j in range(batch_size):
        #     for k in range(max_len):
        #         if temp_my_attention_mask[j, k] == 0:
        #             continue
        #         else:
        #             i = torch.sum(temp_my_attention_mask[j, :k])
        #             new_sentences[j, k] = summed[i, j]
        #
        # # gives dimension [batch size, max length] with 0s at masked tokens and remaining tokens at non masked tokens
        # remaining_sentences = temp_input_ids * (torch.ones(*temp_my_attention_mask.shape) - temp_my_attention_mask)
        #
        # assert remaining_sentences.shape == new_sentences.shape, 'shape of remaining is {} and new is {}'.format(*remaining_sentences.shape, *new_sentences.shape)

        # TODO should output a tensor of dimension [batch size, 4, max length, vocab size] with one hot vectos along the fourth dimension
        out_dict = {k: v for k, v in kwargs.items()}
        # reshape to resemble known form
        out_dict['input_ids'] = input_ids
        out_dict['my_attention_mask'] = my_attention_mask
        out_dict['attention_mask'] = attention_mask
        out_dict['token_type_ids'] = token_type_ids
        out_dict['inputs_embeds'] = onehots

        return out_dict


# def int_list(l):
#     return [int(o) for o in l]
#
#
# def randomize_generator(features, model=None):
#     if model is None:
#         for f in features:
#             assert isinstance(f, ArcFeature)
#
#             cf = f.choices_features
#             for d in cf:
#                 # cf is a list of dicts with keys 'input_ids', ..., ..., 'attention_mask'
#                 new_input_ids = [ii//2 if am == 1 else ii for ii, am in zip(d['input_ids'], d['attention_mask'])]
#                 d['input_ids'] = new_input_ids
#     else:
#         all_input_ids = []
#         for f in features:
#             assert isinstance(f, ArcFeature)
#
#             cf = f.choices_features
#             all_input_ids.extend([d['input_ids'] for d in cf])
#
#         input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.float)
#         output = model(input_ids_tensor)
#
#         output = output.view((len(features), 4, -1))
#
#         for ii, f in enumerate(features):
#             cf = f.choices_features
#             for jj, d in enumerate(cf):
#                 d['input_ids'] = int_list(output[ii, jj, :].tolist())
#
#     return features
#
#
# def generator(features, model, randomize=False):
#     if randomize:
#         return randomize_generator(features, model)


generator_models_and_config_classes = {
    'seq': (GeneratorConfig, Seq2Seq),
    'bert': (BertConfig, BertForMaskedLM),
    'roberta': (RobertaConfig, RobertaForMaskedLM),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM),
    'albert': (AlbertConfig, MyAlbertForMaskedLM)
}

