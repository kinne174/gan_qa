import torch
import torch.nn as nn
from transformers import (BertForMaskedLM, RobertaForMaskedLM,
                          BertConfig, RobertaConfig, AlbertConfig,
                          AlbertForMaskedLM)
from transformers import PretrainedConfig
from torch.nn import functional as F
import logging
from collections import Counter
import math
import os
import json

logger = logging.getLogger(__name__)

# from https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()

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
        super(Decoder, self).__init__()

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
def sample_gumbel(shape, device, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0, 1).to(device)
    # logger.info('Generator device is {}'.format(device))
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, device, temperature):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, device, temperature=0.5, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, device, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y


class Seq2Seq(nn.Module):
    def __init__(self, config, device):
        super(Seq2Seq, self).__init__()

        self.encoder = config.encoder
        self.decoder = config.decoder

        self.device = device

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    @classmethod
    def from_pretrained(cls, config, device):
        return cls(config, device)

    def forward(self, input_ids, my_attention_mask, **kwargs):

        temp_input_ids = torch.t(input_ids.view(-1, input_ids.shape[-1]))  # [max len, batch size]
        temp_my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])  # [batch size, max len]

        batch_size = temp_input_ids.shape[1]  # this should be 4*batch size
        max_len = temp_input_ids.shape[0]  # this should be max length
        out_len = max(torch.sum(temp_my_attention_mask, dim=1))  # number of maximum masked tokens
        vocab_size = self.decoder.output_dim  # this should be the number of possible vocab words

        # tensor to store decoder outputs [max attention masks, batch size, vocab size]
        outputs = torch.rand((out_len, batch_size, vocab_size)).to(self.device)
        # outputs = torch.rand((max_len, batch_size, vocab_size)).to(self.device)

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

            # get the true token to supply next
            input = temp_input_ids[t, :]

        # should give dimension [max attention masks, 4*batch size, vocab size] with one hot vectors along the third dimension
        gs = gumbel_softmax(outputs, self.device, temperature=max(0.5, math.exp(-1*(3e-3)*kwargs['update_step'])), hard=True)

        # should start with dimension [4*batch_size, max length, vocab size] with one hot vectors along the third dimension
        # one hot vectors are indicative of the word ids to be used by the classifier
        onehots = torch.zeros((batch_size, max_len, vocab_size)).to(self.device)
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
        # change to sparse for memory savings...?? not sure if that actually helps
        onehots = onehots.to_sparse()

        out_dict = {k: v for k, v in kwargs.items()}
        # reshape input ids to resemble known form
        out_dict['input_ids'] = input_ids
        out_dict['inputs_embeds'] = onehots
        out_dict['my_attention_mask'] = my_attention_mask

        return out_dict

class Seq2SeqReinforce(nn.Module):
    def __init__(self, config):
        super(Seq2SeqReinforce, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # self.encoder = nn.Sequential()
        # self.encoder.add_module('encoder_embedding', embedding)
        # self.encoder.add_module('encoder_dropout', nn.Dropout(p=0.10))
        self.encoder = nn.LSTM(config.embedding_dim, config.encoder_decoder_hidden_dim,
                                                        batch_first=True, num_layers=config.num_layers,
                                                        dropout=0.10)

        # self.decoder = nn.Sequential()
        # self.decoder.add_module('decoder_embedding', embedding)
        # self.decoder.add_module('decoder_dropout', nn.Dropout(p=0.10))
        self.decoder = nn.LSTM(config.embedding_dim, config.encoder_decoder_hidden_dim,
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
        self.one_hot_of_known_words = None
        self.vocab_size = config.vocab_size

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        model_to_save = self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'model_weights.pt')
        torch.save(model_to_save.state_dict(), output_model_file)

        assert hasattr(model_to_save, "config")
        config_filename = os.path.join(save_directory, 'config.json')
        with open(config_filename, 'w') as cf:
            json.dump(vars(model_to_save.config), cf)

        logger.info("Model weights and config saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        if pretrained_model_name_or_path is not None and os.path.exists(pretrained_model_name_or_path):
            config_filename = os.path.join(pretrained_model_name_or_path, 'config.json')
            with open(config_filename, 'r') as cf:
                config_json = json.load(cf)
            config = GeneratorConfig(**config_json)

            model_to_return = cls(config)
            model_load_filename = os.path.join(pretrained_model_name_or_path, 'model_weights.pt')
            model_to_return.load_state_dict(torch.load(model_load_filename))

            return model_to_return

        return cls(config)

    def set_known_words_masking(self, input_ids):
        counter = Counter(input_ids.view(-1).cpu().numpy())
        for _ in range(3):
            counter.subtract(counter.keys())

        counter += Counter()

        one_hot_of_known_words = torch.zeros(self.vocab_size, dtype=torch.float).scatter_(0, torch.tensor(list(counter.keys()), dtype=torch.long), 1)

        logger.info('The new vocabulary has size {}.'.format(len(counter)))

        setattr(self, 'one_hot_of_known_words', one_hot_of_known_words)

    def forward(self, input_ids, attention_mask, token_type_ids, classification_labels):

        # input ids is [batch size, 4, seq length]
        # batch_size = input_ids.shape[0]

        # input ids is [4*batch size, seq length]
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        # embedded_ids is [4*batch size, seq length, embedding dim]
        embbeded_ids = self.embedding(input_ids)

        _, (hidden, cell) = self.encoder(embbeded_ids)

        # initial decoder input is [4*batch size, 1]
        decoder_input = input_ids[:, 0].unsqueeze(-1)

        output_prediction_scores = None

        for t in range(1, input_ids.shape[-1]):
            # embedded_input is [4*batch size, t, embedding dim]
            embedded_input = self.embedding(decoder_input)

            # decoder output is [4*batch size, 1, hidden dim]
            decoder_output, (hidden, cell) = self.decoder(embedded_input, (hidden, cell))

            # decoder output is [4*batch size, hidden dim]
            decoder_output = decoder_output[:, -1, :]

            # classifier states is [4*batch size, classifier layer size]
            classifier_states = self.classification_layer(classification_labels.view(-1, 1))

            # classifier states is [batch size, 4, classifier layer size]
            # classifier_states = classifier_states.view(-1, 4, classifier_states.shape[-1])

            # decoder classifier is [4*batch size, hidden dim + classifier layer size]
            decoder_classifier_output = torch.cat((decoder_output, classifier_states), dim=-1)

            # decoder prediction is [4*batch size, vocab size]
            decoder_prediction = self.fully_connected(decoder_classifier_output)

            # decoder prediction is [4*batch size, 1, vocab size]
            decoder_prediction = decoder_prediction.unsqueeze(1)

            # output predictions is [4*batch size, t, vocab size]
            if output_prediction_scores is None:
                output_prediction_scores = decoder_prediction
            else:
                output_prediction_scores = torch.cat((output_prediction_scores, decoder_prediction), dim=1)

            # next inputs is [4*batch size]
            next_inputs = torch.where(input_ids[:, t] == self.mask_id, torch.argmax(decoder_prediction.squeeze(), dim=-1), input_ids[:, t])

            # decoder input is [4*batch size, t+1]
            decoder_input = torch.cat((decoder_input, next_inputs.unsqueeze(-1)), dim=-1)

        # output prediction scores is [batch size, 4, seq length-1, vocab size]
        output_prediction_scores = output_prediction_scores.view(-1, 4, *output_prediction_scores.shape[1:])

        # output prediction scores is [batch size, 4, seq length, vocab size]
        output_prediction_scores = torch.cat((torch.zeros_like(output_prediction_scores[..., 0, :]).unsqueeze(-2), output_prediction_scores), dim=-2)

        if self.one_hot_of_known_words is not None:
            output_prediction_scores_cpu = output_prediction_scores.cpu()
            output_prediction_scores_cpu = torch.where(self.one_hot_of_known_words.expand_as(output_prediction_scores_cpu) == 1,
                                                   output_prediction_scores_cpu, torch.ones_like(output_prediction_scores_cpu) * (-2)**31)
            output_prediction_scores = output_prediction_scores_cpu.to(input_ids.device)
            # self.one_hot_of_known_words = self.one_hot_of_known_words.to(output_prediction_scores.device)
            # output_prediction_scores = torch.where(self.one_hot_of_known_words.expand_as(output_prediction_scores) == 1,
            #                                        output_prediction_scores, torch.ones_like(output_prediction_scores) * (-2)**31)

        # log softmax of scores
        output_prediction_scores = self.log_softmax(output_prediction_scores)

        return output_prediction_scores


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
        return cls(**kwargs)


class GeneralModelforMaskedLM(nn.Module):
    def __init__(self):
        super(GeneralModelforMaskedLM, self).__init__()

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)

    def forward(self, input_ids, my_attention_mask, attention_mask, **kwargs):

        if hasattr(self.model, 'roberta'):
            token_type_ids = None
        elif hasattr(self.model, 'albert'):
            token_type_ids = kwargs['token_type_ids']
        else:
            raise NotImplementedError

        # change from dimension [batch size, 4, max length] to [4*batch size, max length]
        temp_input_ids = input_ids.view(-1, input_ids.shape[-1])
        temp_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        temp_my_attention_mask = my_attention_mask.view(-1, my_attention_mask.shape[-1])  # [batch size, max len]

        batch_size = temp_input_ids.shape[0]  # this should be 4*batch size
        max_len = temp_input_ids.shape[1]  # this should be max length
        out_len = max(torch.sum(temp_my_attention_mask, dim=1))  # number of maximum masked tokens

        assert temp_input_ids.dtype == torch.long
        # outputs dimension [4*batch size, max length, vocab size] of before softmax scores for each word
        model_outputs = self.model(input_ids=temp_input_ids,
                                   attention_mask=temp_attention_mask,
                                   token_type_ids=token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None)

        prediction_scores = model_outputs[0]
        vocab_size = prediction_scores.shape[-1]

        # tensor to store decoder outputs [max attention masks, batch size, vocab size]
        outputs = torch.rand((out_len, batch_size, vocab_size)).to(self.device)

        for t in range(1, max_len):
            # place predictions in a tensor holding predictions for each token
            for j in range(batch_size):
                if temp_my_attention_mask[j, t] == 0:
                    continue
                else:
                    i = torch.sum(temp_my_attention_mask[j, :t])
                    outputs[i, j, :] = prediction_scores[j, t, :]

        # should give dimension [max attention masks, 4*batch size, vocab size] with one hot vectors along the third dimension
        gs = gumbel_softmax(outputs, self.device, hard=True)

        # should start with dimension [4*batch_size, max length, vocab size] with one hot vectors along the third dimension
        # one hot vectors are indicative of the word ids to be used by the classifier
        onehots = torch.zeros((batch_size, max_len, vocab_size)).to(self.device)
        onehot = torch.FloatTensor(1, vocab_size)
        for i in range(batch_size):
            for j in range(max_len):
                if temp_my_attention_mask[i, j] == 1:
                    k = torch.sum(temp_my_attention_mask[i, :j])
                    onehots[i, j, :] = gs[k, i, :]
                else:
                    onehot.zero_()
                    onehot.scatter_(1, torch.tensor([temp_input_ids[i, j]]).unsqueeze(0), 1)
                    onehots[i, j, :] = onehot

        # change to dimension [4*batch size*max length, vocab size] to make multiplying by embeddings in classifier easier
        onehots = onehots.view(-1, onehots.shape[-1])
        # change to sparse for memory savings...?? not sure if that actually helps
        # onehots = onehots.to_sparse()

        # should output a tensor of dimension [batch size, 4, max length, vocab size] with one hot vectos along the fourth dimension
        out_dict = {k: v for k, v in kwargs.items()}
        # reshape to resemble known form
        out_dict['input_ids'] = input_ids
        out_dict['my_attention_mask'] = my_attention_mask
        out_dict['attention_mask'] = attention_mask
        # out_dict['token_type_ids'] = token_type_ids
        out_dict['inputs_embeds'] = onehots

        return out_dict


class MyAlbertForMaskedLM(GeneralModelforMaskedLM):
    def __init__(self, pretrained_model_name_or_path, config, device):
        super(MyAlbertForMaskedLM, self).__init__()
        self.model = AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config)
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, device):
        return cls(pretrained_model_name_or_path, config, device)


class MyRobertaForMaskedLM(GeneralModelforMaskedLM):
    def __init__(self, pretrained_model_name_or_path, config, device):
        super(MyRobertaForMaskedLM, self).__init__()
        self.model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config)
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, device):
        return cls(pretrained_model_name_or_path, config, device)


class ElementwiseMultiplication(nn.Module):
    def __init__(self, num_features, bias):
        super(ElementwiseMultiplication, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(0, math.sqrt(5))
            if self.bias is not None:
                self.bias.uniform_(-1*math.sqrt(5), math.sqrt(5))

    def forward(self, x):
        out = x * self.weight
        if self.bias is not None:
            out += self.bias
        return out


class Weight_Clipper(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            w = module.weight.data
            w = w.clamp(-1., 1.)
            module.weight.data = w

        if hasattr(module, 'bias') and module.bias is not None:
                b = module.bias.data
                b = b.clamp(1e-12, 1.)
                module.bias.data = b


class GeneratorReinforcement(nn.Module):
    def __init__(self, model):
        super(GeneratorReinforcement, self).__init__()

        assert hasattr(self, 'classification_layer_size')
        assert hasattr(self, 'temperature')
        assert hasattr(self, 'is_oracleM')

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.model = model

        self.vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        self.classification_layer = nn.Linear(1, self.classification_layer_size, bias=True)
        self.decoder = nn.Sequential()
        self.decoder.add_module('linear1', nn.Linear(hidden_size + self.classification_layer_size, hidden_size + self.classification_layer_size))
        self.decoder.add_module('relu', nn.ReLU())
        self.decoder.add_module('norm', nn.LayerNorm(hidden_size + self.classification_layer_size))
        self.decoder.add_module('linear2', nn.Linear(hidden_size + self.classification_layer_size, self.vocab_size))

        self.one_hot_of_known_words = None

    def set_known_words_masking(self, input_ids):
        counter = Counter(input_ids.view(-1).cpu().numpy())
        for _ in range(3):
            counter.subtract(counter.keys())

        counter += Counter()

        one_hot_of_known_words = torch.zeros(self.vocab_size, dtype=torch.float).scatter_(0, torch.tensor(list(counter.keys()), dtype=torch.long), 1)

        logger.info('The new vocabulary has size {}.'.format(len(counter)))

        setattr(self, 'one_hot_of_known_words', one_hot_of_known_words)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        self.model.save_pretrained(save_directory)

        model_to_save = self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'model_weights.pt')
        torch.save(model_to_save.state_dict(), output_model_file)

    def forward(self, input_ids, attention_mask, token_type_ids, classification_labels):
        if hasattr(self.model, 'roberta'):
            token_type_ids = None
        elif hasattr(self.model, 'albert'):
            pass
        elif hasattr(self.model, 'bert'):
            pass
            # idx = 1
            # input_ids_to_select = torch.tensor([[1], [2]], dtype=torch.long).to(classification_labels.device)
            # input_ids_to_insert = torch.index_select(input=input_ids_to_select, dim=0, index=classification_labels.long().view(-1)).view(*classification_labels.shape, input_ids_to_select.shape[1])
            # input_ids = torch.cat((input_ids[..., :idx], input_ids_to_insert, input_ids[..., idx:]), dim=-1)
            #
            # token_type_ids = torch.cat((token_type_ids[..., :idx], torch.zeros_like(input_ids_to_insert), token_type_ids[..., idx:]), dim=-1)
            # attention_mask = torch.cat((attention_mask[..., :idx], torch.ones_like(input_ids_to_insert), attention_mask[..., idx:]), dim=-1)

        else:
            raise NotImplementedError

        # change from dimension [batch size, 4, max length] to [4*batch size, max length]
        temp_input_ids = input_ids.view(-1, input_ids.shape[-1])
        temp_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        assert temp_input_ids.dtype == torch.long
        # outputs dimension [4*batch size, max length, vocab size] of before softmax scores for each word
        model_outputs = self.model(input_ids=temp_input_ids,
                                   attention_mask=temp_attention_mask,
                                   token_type_ids=token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None)

        prediction_scores = model_outputs[0]

        if not self.is_oracleM:
            hidden_states = model_outputs[1][-1]

            classifier_states = self.classification_layer(classification_labels.view(-1, 1))
            classifier_states = torch.cat([classifier_states.unsqueeze(1)] * hidden_states.shape[1], dim=1)

            hidden_states = torch.cat((hidden_states, classifier_states), dim=-1)

            prediction_scores = self.decoder(hidden_states)

        if self.one_hot_of_known_words is not None:
            self.one_hot_of_known_words = self.one_hot_of_known_words.to(prediction_scores.device)
            # prediction_scores *= (((-2 ** 31) * (1 - self.one_hot_of_known_words.view(1, 1, -1))) + self.one_hot_of_known_words.view(1, 1, -1))
            prediction_scores = torch.where(self.one_hot_of_known_words.expand_as(prediction_scores) == 1, prediction_scores, torch.ones_like(prediction_scores) * (-2)**31)

        prediction_scores = prediction_scores.view(-1, 4, *prediction_scores.shape[1:])

        temperature = 1. if not self.training else self.temperature
        prediction_scores = self.log_softmax(prediction_scores / temperature)

        return prediction_scores


class MyRobertaForMaskedLMReinforcement(GeneratorReinforcement):
    def __init__(self, pretrained_model_name_or_path, config):

        for k, v in config.task_specific_params.items():
            setattr(self, k, v)

        super(MyRobertaForMaskedLMReinforcement, self).__init__(model=RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        model_to_return = cls(pretrained_model_name_or_path, config)
        if pretrained_model_name_or_path is not None and os.path.exists(os.path.join(pretrained_model_name_or_path, 'model_weights.pt')):
            model_load_filename = os.path.join(pretrained_model_name_or_path, 'model_weights.pt')
            model_to_return.load_state_dict(torch.load(model_load_filename))

        return model_to_return


class MyBertForMaskedLMReinforcement(GeneratorReinforcement):
    def __init__(self, pretrained_model_name_or_path, config):

        for k, v in config.task_specific_params.items():
            setattr(self, k, v)

        super(MyBertForMaskedLMReinforcement, self).__init__(model=BertForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        model_to_return = cls(pretrained_model_name_or_path, config)
        if pretrained_model_name_or_path is not None and os.path.exists(os.path.join(pretrained_model_name_or_path, 'model_weights.pt')):
            model_load_filename = os.path.join(pretrained_model_name_or_path, 'model_weights.pt')
            model_to_return.load_state_dict(torch.load(model_load_filename))

        return model_to_return

generator_models_and_config_classes = {
    'seq': (GeneratorConfig, Seq2Seq),
    'roberta': (RobertaConfig, MyRobertaForMaskedLM),
    'albert': (AlbertConfig, MyAlbertForMaskedLM),
    'roberta-reinforce': (RobertaConfig, MyRobertaForMaskedLMReinforcement),
    'bert-reinforce': (BertConfig, MyBertForMaskedLMReinforcement),
    'seq-reinforce': (GeneratorConfig, Seq2SeqReinforce),
}

