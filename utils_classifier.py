import numpy as np
import torch
import torch.nn as nn
from utils_embedding_model import ArcFeature
from transformers import (BertForMultipleChoice, RobertaForMultipleChoice, XLMRobertaForMultipleChoice,
                          BertConfig, RobertaConfig, XLMRobertaConfig, AlbertPreTrainedModel, AlbertConfig,
                          AlbertModel)
from transformers import PretrainedConfig
import logging

logger = logging.getLogger(__name__)

# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_device()


class ClassifierConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super(ClassifierConfig, self).__init__()

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls(**kwargs)


class ClassifierNet(torch.nn.Module):

    def __init__(self, config):
        super(ClassifierNet, self).__init__()
        self.num_choices = config.num_choices
        self.in_features = config.in_features
        self.hidden_features = config.hidden_features

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dimension, padding_idx=0)

        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=True),
            nn.BatchNorm1d(self.hidden_features),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Linear(self.hidden_features, 1),
            nn.Sigmoid()
        )

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    @classmethod
    def from_pretrained(cls, config):
        return cls(config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels, **kwargs):
        if 'inputs_embeds' in kwargs:
            # embedding matrix should be of dimension [vocab size, embedding dimension]
            embeddings = self.embedding.weight.to(device)
            # input_embeds should be of dimension [4*batch size*max length, vocab size]
            input_embeds = kwargs['inputs_embeds']
            assert input_embeds.is_sparse
            temp_input_ids = torch.sparse.mm(input_embeds, embeddings)
            temp_input_ids = temp_input_ids.view(*input_ids.shape, -1)
            temp_input_ids = torch.mean(temp_input_ids, dim=-1)
            temp_input_ids = temp_input_ids.view(-1, temp_input_ids.shape[-1])
        else:
            temp_input_ids = input_ids.view(-1, input_ids.shape[-1])

        x = self.linear(temp_input_ids.float())
        x = self.out(x)

        x = x.view(-1, self.num_choices)

        if labels is not None:
            loss = self.BCEWithLogitsLoss(x, labels)
            return x, loss

        return x, None


def flip_labels(classification_labels, discriminator_labels, **kwargs):

    out_c_labels = torch.ones(*classification_labels.shape).to(device) - classification_labels
    out_d_labels = torch.ones(*discriminator_labels.shape).to(device) - discriminator_labels

    out_dict = {k: v for k, v in kwargs.items()}
    out_dict['classification_labels'] = out_c_labels
    out_dict['discriminator_labels'] = out_d_labels

    return out_dict


class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        if input_ids is not None:
            num_choices = input_ids.shape[1]
            input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            num_choices = inputs_embeds.shape[1]
            inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[2], inputs_embeds.shape[3])

        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class MyAlbertForMultipleChoice(nn.Module):
    def __init__(self, pretrained_model_name_or_path, config):
        super(MyAlbertForMultipleChoice, self).__init__()
        self.albert = AlbertForMultipleChoice.from_pretrained(pretrained_model_name_or_path, config=config)

        self.discriminator = nn.Linear(self.albert.config.hidden_size, 1)


        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(pretrained_model_name_or_path, config)

    def save_pretrained(self, save_directory):
        return self.albert.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask, token_type_ids, classification_labels, discriminator_labels, **kwargs):

        if 'inputs_embeds' in kwargs:
            embeddings = self.albert.albert.embeddings.word_embeddings.weight.to(device)
            inputs_embeds = kwargs['inputs_embeds']
            assert inputs_embeds.is_sparse
            temp_inputs_embeds = torch.sparse.mm(inputs_embeds, embeddings)
            temp_inputs_embeds = temp_inputs_embeds.view(*input_ids.shape, -1)

            outputs = self.albert(token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=temp_inputs_embeds)
        else:
            outputs = self.albert(input_ids=input_ids.long(),
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)

        last_hidden_state = outputs[1][0]
        discriminator_scores = self.discriminator(last_hidden_state)

        classification_scores = outputs[0]

        scores = (classification_scores, discriminator_scores)

        if discriminator_labels is not None:

            assert discriminator_scores.shape == discriminator_labels.shape, 'Discriminator shape ({}) is not the same as labels shape ({})'.format(discriminator_scores.shape,
                                                                                                                                                    discriminator_labels.shape)

            discriminator_loss = self.BCEWithLogitsLoss(discriminator_scores, discriminator_labels)

            if classification_labels is not None:

                assert classification_scores.shape == classification_labels.shape, 'classification shape is {} and labels shape is {}'.format(classification_scores.shape,
                                                                                                                                          classifier_labels.shape)
                classification_loss = self.BCEWithLogitsLoss(classification_scores, classification_labels)

            else:
                classification_loss = None

            loss = (classification_loss, discriminator_loss)
        else:
            loss = (None, None)

        return scores, loss

class MyBertForMultipleChoice(nn.Module):
    def __init__(self, pretrained_model_name_or_path, config):
        super(MyBertForMultipleChoice, self).__init__()
        self.bert = BertForMultipleChoice.from_pretrained(pretrained_model_name_or_path, config=config)

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(pretrained_model_name_or_path, config)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, **kwargs):

        outputs = self.bert(input_ids=input_ids.long(),
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)
        classification_scores = outputs[0]

        if labels is not None:

            assert classification_scores.shape == labels.shape, 'classification shape is {} and labels shape is {}'.format(classification_scores.shape,
                                                                                                                           labels.shape)
            loss = self.BCEWithLogitsLoss(classification_scores, labels)
            return classification_scores, loss

        return classification_scores, None


classifier_models_and_config_classes = {
    'linear': (ClassifierConfig, ClassifierNet),
    'bert': (BertConfig, MyBertForMultipleChoice),
    'roberta': (RobertaConfig, RobertaForMultipleChoice),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForMultipleChoice),
    'albert': (AlbertConfig, MyAlbertForMultipleChoice),
}