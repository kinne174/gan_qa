import numpy as np
import torch
import torch.nn as nn
from transformers import (BertForMultipleChoice, RobertaForMultipleChoice, XLMRobertaForMultipleChoice,
                          BertConfig, RobertaConfig, XLMRobertaConfig, AlbertPreTrainedModel, AlbertConfig,
                          AlbertModel)
from transformers import PretrainedConfig
import logging
import os

logger = logging.getLogger(__name__)


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


class GeneralModelForMultipleChoice(nn.Module):
    def __init__(self, model):
        super(GeneralModelForMultipleChoice, self).__init__()

        self.model = model
        self.discriminator = nn.Sequential(nn.BatchNorm1d(self.model.config.hidden_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.model.config.hidden_size, 1),
                                           nn.Tanh())

        self.BCEWithLogitsLoss_noreduc = nn.BCEWithLogitsLoss(reduction='none')
        self.BCEWithLogitsLoss_reduc = nn.BCEWithLogitsLoss(reduction='mean')

    @staticmethod
    def Wloss(preds, labels):
        return torch.mean(preds * labels)

    def save_pretrained(self, save_directory):
        return self.model.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask, token_type_ids, classification_labels, discriminator_labels,
                sentences_type, **kwargs):
        batch_size = input_ids.shape[0]

        if 'inputs_embeds' in kwargs:
            embeddings = self.albert.albert.embeddings.word_embeddings.weight.to(self.device)
            inputs_embeds = kwargs['inputs_embeds']
            assert inputs_embeds.is_sparse
            temp_inputs_embeds = torch.sparse.mm(inputs_embeds, embeddings)
            temp_inputs_embeds = temp_inputs_embeds.view(*input_ids.shape, -1)

            outputs = self.model(token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=temp_inputs_embeds)
        else:
            outputs = self.model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)

        last_cls_hidden_state = outputs[1][0][:, 0, :].squeeze().view(batch_size * 4, -1)
        discriminator_scores = self.discriminator(last_cls_hidden_state)
        # discriminator_scores = F.softmax(discriminator_scores.squeeze(), dim=1)

        classification_scores = outputs[0]
        # classification_scores = F.softmax(classification_scores, dim=1)

        scores = (classification_scores, discriminator_scores)

        if discriminator_labels is not None:

            # assert xd.shape == discriminator_labels.shape, 'Discriminator shape ({}) is not the same as labels shape ({})'.format(xd.shape,
            #                                                                                                                       discriminator_labels.shape)

            discriminator_loss = self.Wloss(discriminator_scores,
                                            discriminator_labels.view(*discriminator_scores.shape))

            if classification_labels is not None and not torch.all(
                    torch.eq(torch.zeros_like(sentences_type), sentences_type)):

                assert classification_scores.shape == classification_labels.shape, 'classification shape is {} and labels shape is {}'.format(
                    classification_scores.shape,
                    classification_labels.shape)
                classification_loss_noreduc = self.BCEWithLogitsLoss_noreduc(classification_scores,
                                                                             classification_labels)

                sentences_type_multiplier = sentences_type.unsqueeze(1) * torch.ones_like(classification_loss_noreduc)

                assert classification_loss_noreduc.shape == sentences_type_multiplier.shape, 'classification loss shape ({}) is not the same as sentences_type shape ({})'.format(
                    classification_loss_noreduc.shape,
                    sentences_type.shape)

                classification_loss = torch.sum(classification_loss_noreduc * sentences_type_multiplier) / (
                    torch.sum(sentences_type_multiplier))

            else:
                classification_loss = None

            loss = (classification_loss, discriminator_loss)
        else:
            loss = (None, None)

        return scores, loss


class ClassifierNet(nn.Module):

    def __init__(self, config, device):
        super(ClassifierNet, self).__init__()
        self.num_choices = config.num_choices
        self.in_dim = config.in_features
        self.hidden_dim = config.hidden_features

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dimension, padding_idx=0)

        self.linear = nn.Linear(self.in_dim, self.hidden_dim, bias=True)

        self.out_classification = nn.Sequential(nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(self.hidden_dim, 1))
        self.out_discriminator = nn.Sequential(nn.BatchNorm1d(self.hidden_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(self.hidden_dim, 1))

        self.gru = nn.GRU(config.embedding_dimension, self.hidden_dim, batch_first=True, num_layers=2, dropout=0.1)

        self.BCEWithLogitsLoss_noreduc = nn.BCEWithLogitsLoss(reduction='none')
        self.BCEWithLogitsLoss_reduc = nn.BCEWithLogitsLoss(reduction='mean')

        self.device = device

    @staticmethod
    def Wloss(preds, labels):
        return torch.mean(preds*labels)

    @classmethod
    def from_pretrained(cls, **kwargs):
        pretrained_model_name_or_path = kwargs['pretrained_model_name_or_path']

        if pretrained_model_name_or_path is not None:
            logger.info('Attempting to load model from checkpoint {}'.format(pretrained_model_name_or_path))

            if os.path.exists(pretrained_model_name_or_path):
                logger.info('Checkpoint found! Loading pretrained model.')

                model_to_return = cls(kwargs['config'], kwargs['device'])
                model_load_filename = os.path.join(pretrained_model_name_or_path, 'linear_weights.pt')
                model_to_return.load_state_dict(torch.load(model_load_filename))

                return model_to_return

            else:
                logger.info('Unable to load model. Returning new model.')

        return cls(kwargs['config'], kwargs['device'])

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        model_to_save = self

        # TODO save dimensions of embedding and hidden

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'linear_weights.pt')
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def forward(self, input_ids, attention_mask, token_type_ids, classification_labels, discriminator_labels, sentences_type, **kwargs):

        if 'inputs_embeds' in kwargs:
            inputs_embeds = kwargs['inputs_embeds']

            # embedding matrix should be of dimension [vocab size, embedding dimension]
            embedding_mat = self.embedding.weight.to(self.device)

            # input_embeds should be of dimension [4*batch size*max length, vocab size]
            assert inputs_embeds.is_sparse

            temp_input_ids = torch.sparse.mm(inputs_embeds, embedding_mat)
            temp_input_ids = temp_input_ids.view(input_ids.shape[0] * 4, input_ids.shape[-1], -1)
        else:
            temp_input_ids = self.embedding(input_ids.view(-1, input_ids.shape[-1]))

        sum_attentions = torch.sum(attention_mask.view(-1, attention_mask.shape[-1]), dim=1) - 1
        x = torch.empty((input_ids.shape[0] * 4, self.hidden_dim)).to(self.device)

        gru_out, last_hidden = self.gru(temp_input_ids)

        assert sum_attentions.shape[0] == x.shape[0], 'Sum_attentions shape is ({}) not equal to x shape ({})'.format(sum_attentions.shape[0], x.shape[0])
        for sa_ind, sa in enumerate(sum_attentions):
            x[sa_ind, :] = gru_out[sa_ind, sa, :].squeeze()

        # x = last_hidden[-1, :, :].squeeze()

        xc = self.out_classification(x)
        xd = self.out_discriminator(x)

        xc = xc.view(-1, self.num_choices)
        # xc = F.softmax(xc, dim=1)

        # xd = xd.view(-1, self.num_choices)
        # xd = F.softmax(xd, dim=1)

        scores = (xc, xd)

        if discriminator_labels is not None:

            # assert xd.shape == discriminator_labels.shape, 'Discriminator shape ({}) is not the same as labels shape ({})'.format(xd.shape,
            #                                                                                                                       discriminator_labels.shape)

            discriminator_loss = self.Wloss(xd, discriminator_labels.view(*xd.shape))

            if classification_labels is not None and not torch.all(torch.eq(torch.zeros_like(sentences_type), sentences_type)):

                assert xc.shape == classification_labels.shape, 'classification shape is {} and labels shape is {}'.format(xc.shape,
                                                                                                                           classification_labels.shape)
                classification_loss_noreduc = self.BCEWithLogitsLoss_noreduc(xc, classification_labels)

                sentences_type_multiplier = sentences_type.unsqueeze(1)*torch.ones_like(classification_loss_noreduc)

                assert classification_loss_noreduc.shape == sentences_type_multiplier.shape, 'classification loss shape ({}) is not the same as sentences_type shape ({})'.format(classification_loss_noreduc.shape,
                                                                                                                                                                                  sentences_type.shape)

                classification_loss = torch.sum(classification_loss_noreduc*sentences_type_multiplier)/(torch.sum(sentences_type_multiplier))

            else:
                classification_loss = None

            loss = (classification_loss, discriminator_loss)
        else:
            loss = (None, None)

        return scores, loss


def flip_labels(classification_labels, discriminator_labels, **kwargs):

    out_c_labels = torch.ones_like(classification_labels) - classification_labels
    out_d_labels = -1*torch.ones_like(discriminator_labels)

    out_dict = {k: v for k, v in kwargs.items()}
    out_dict['classification_labels'] = out_c_labels
    out_dict['discriminator_labels'] = out_d_labels

    return out_dict


class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForMultipleChoice, self).__init__(config)

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


class MyAlbertForMultipleChoice(GeneralModelForMultipleChoice):
    def __init__(self, pretrained_model_name_or_path, config, device):
        super(MyAlbertForMultipleChoice, self).__init__(model=AlbertForMultipleChoice.from_pretrained(pretrained_model_name_or_path, config=config))
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, device):
        return cls(pretrained_model_name_or_path, config, device)


class MyRobertForMultipleChoice(GeneralModelForMultipleChoice):
    def __init__(self, pretrained_model_name_or_path, config, device):
        super(MyRobertForMultipleChoice, self).__init__(model=RobertaForMultipleChoice.from_pretrained(pretrained_model_name_or_path, config=config))
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, device):
        return cls(pretrained_model_name_or_path, config, device)


classifier_models_and_config_classes = {
    'linear': (ClassifierConfig, ClassifierNet),
    'roberta': (RobertaConfig, MyRobertForMultipleChoice),
    'albert': (AlbertConfig, MyAlbertForMultipleChoice),
}