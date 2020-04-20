import torch
import torch.nn as nn
from transformers import (BertPreTrainedModel, RobertaConfig, AlbertPreTrainedModel, AlbertConfig, AlbertModel, RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
from transformers import PretrainedConfig
import logging
import os
import json

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

        self.BCEWithLogitsLoss_noreduc = nn.BCEWithLogitsLoss(reduction='none')

    @staticmethod
    def my_CrossEntropyLoss(scores, labels):
        if labels.nonzero().shape[0] <= labels.shape[0]:
            # non generated
            return -1*torch.log(torch.exp(scores[labels.nonzero(as_tuple=True)])/torch.sum(torch.exp(scores), dim=-1))
        else:
            # generated
            return torch.log(torch.exp(scores[(labels == 0).nonzero(as_tuple=True)])/torch.sum(torch.exp(scores), dim=-1))

    def save_pretrained(self, save_directory):
        return self.model.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask, token_type_ids, classification_labels,
                sentences_type, **kwargs):

        if torch.all(torch.eq(torch.zeros_like(sentences_type), sentences_type)):
            return None, None

        if hasattr(self.model, 'roberta'):
            token_type_ids = None
        elif hasattr(self.model, 'albert'):
            pass
        else:
            raise NotImplementedError

        if 'inputs_embeds' in kwargs:
            if hasattr(self.model, 'roberta'):
                embeddings = self.model.roberta.embeddings.word_embeddings.weight.to(self.device)
            elif hasattr(self.model, 'albert'):
                embeddings = self.model.albert.embeddings.word_embeddings.weight.to(self.device)
            else:
                raise NotImplementedError

            inputs_embeds = kwargs['inputs_embeds']
            if inputs_embeds.is_sparse:
                temp_inputs_embeds = torch.sparse.mm(inputs_embeds, embeddings)
            else:
                temp_inputs_embeds = torch.mm(inputs_embeds, embeddings)

            temp_inputs_embeds = temp_inputs_embeds.view(*input_ids.shape, -1)

            outputs = self.model(token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=temp_inputs_embeds)
        else:
            outputs = self.model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)

        classification_scores = outputs[0]
        # classification_scores = F.softmax(classification_scores, dim=1)

        scores = classification_scores

        if classification_labels is not None:

            assert classification_scores.shape == classification_labels.shape, 'classification shape is {} and labels shape is {}'.format(
                classification_scores.shape,
                classification_labels.shape)
            classification_loss_noreduc = self.my_CrossEntropyLoss(classification_scores,
                                                                   classification_labels)

            # sentences_type_multiplier = sentences_type.unsqueeze(1) * torch.ones_like(classification_loss_noreduc)

            assert classification_loss_noreduc.shape == sentences_type.shape, 'classification loss shape ({}) is not the same as sentences_type shape ({})'.format(
                classification_loss_noreduc.shape,
                sentences_type.shape)

            classification_loss = torch.sum(classification_loss_noreduc * sentences_type) / (
                torch.sum(sentences_type))

        else:
            classification_loss = None

        loss = classification_loss

        return scores, loss

def flip_labels(classification_labels, discriminator_labels, **kwargs):

    out_c_labels = torch.ones_like(classification_labels) - classification_labels
    out_d_labels = discriminator_labels * -1

    out_dict = {k: v for k, v in kwargs.items()}
    out_dict['classification_labels'] = out_c_labels
    out_dict['discriminator_labels'] = out_d_labels

    return out_dict


class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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

        outputs = self.roberta(
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


class GeneralModelForMultipleChoiceReinforce(nn.Module):
    def __init__(self, model):
        super(GeneralModelForMultipleChoiceReinforce, self).__init__()

        self.model = model

    def save_pretrained(self, save_directory):
        return self.model.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask, token_type_ids):

        if hasattr(self.model, 'roberta'):
            token_type_ids = None
        elif hasattr(self.model, 'albert'):
            pass
        else:
            raise NotImplementedError

        outputs = self.model(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        logits = outputs[0]

        return logits


class MyRobertForMultipleChoiceReinforce(GeneralModelForMultipleChoiceReinforce):
    def __init__(self, pretrained_model_name_or_path, config):
        super(MyRobertForMultipleChoiceReinforce, self).__init__(model=RobertaForMultipleChoice.from_pretrained(pretrained_model_name_or_path, config=config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(pretrained_model_name_or_path, config)


class ClassifierNetReinforce(nn.Module):

    def __init__(self, config):
        super(ClassifierNetReinforce, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dimension, padding_idx=0)

        self.LSTM = nn.Sequential()
        self.LSTM.add_module('lstm_transformation', nn.Linear(config.embedding_dimension, config.hidden_dim))
        self.LSTM.add_module('lstm', nn.LSTM(config.hidden_dim, config.hidden_dim, batch_first=True, bidirectional=True,
                            num_layers=config.num_layers, dropout=0.10))

        self.MLP = nn.Sequential()
        self.MLP.add_module('linear1', nn.Linear(config.hidden_dim, 128))
        self.MLP.add_module('relu', nn.ReLU())
        self.MLP.add_module('dropout', nn.Dropout(p=0.10))
        self.MLP.add_module('linear2', nn.Linear(128, 1))
        # self.MLP.add_module('sigmoid', nn.Sigmoid())

    @classmethod
    def from_pretrained(cls, config, pretrained_model_name_or_path):
        if pretrained_model_name_or_path is not None and os.path.exists(pretrained_model_name_or_path):
            config_filename = os.path.join(pretrained_model_name_or_path, 'config.json')
            with open(config_filename, 'r') as cf:
                config_json = json.load(cf)
            config = ClassifierConfig(**config_json)

            model_to_return = cls(config)
            model_load_filename = os.path.join(pretrained_model_name_or_path, 'model_weights.pt')
            model_to_return.load_state_dict(torch.load(model_load_filename))

            return model_to_return

        return cls(config)

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

    def forward(self, input_ids, attention_mask, token_type_ids):

        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        input_embeddings = self.embedding(input_ids)

        lstm_out, (last_hidden, last_cell) = self.LSTM(input_embeddings)

        if self.LSTM.lstm.bidirectional:
            lstm_out = torch.mean(lstm_out.view(input_embeddings.shape[0], input_embeddings.shape[1],
                                            2, -1), dim=2)
        lstm_out_no_padding = torch.cat([lstm_out[i, torch.sum(attention_mask[i, :]-1, dtype=torch.long), :].unsqueeze(0) for i in range(input_ids.shape[0])], dim=0)

        logits = self.MLP(lstm_out_no_padding)
        logits = logits.view(-1, 4)

        return logits


classifier_models_and_config_classes = {
    'roberta': (RobertaConfig, MyRobertForMultipleChoice),
    'albert': (AlbertConfig, MyAlbertForMultipleChoice),
    'roberta-reinforce': (RobertaConfig, MyRobertForMultipleChoiceReinforce),
    'linear-reinforce': (ClassifierConfig, ClassifierNetReinforce)
}