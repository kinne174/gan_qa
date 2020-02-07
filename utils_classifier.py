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
        temp_input_ids = input_ids.view(-1, input_ids.shape[-1])
        x = self.linear(temp_input_ids.float())
        x = self.out(x)

        x = x.view(-1, self.num_choices)

        if labels is not None:
            loss = self.BCEWithLogitsLoss(x, labels)
            return x, loss

        return x, None


def flip_labels(labels, **kwargs):
    out_labels = torch.ones(*labels.shape) - labels
    out_dict = {k: v for k, v in kwargs.items() if not k in ['labels']}
    out_dict['labels'] = out_labels
    return out_dict


def randomize_classifier(features, model=None, labels=None):

    if model is None or labels is None:
        return torch.rand((1,)).item(), torch.randn((len(features), 4))
    else:
        input_ids = []
        # labels = torch.tensor(labels, dtype=torch.float)
        for f in features:
            assert isinstance(f, ArcFeature)

            cf = f.choices_features
            input_ids.extend([d['input_ids'] for d in cf])

        input_ids = torch.tensor(input_ids, dtype=torch.float)

        inputs = {'x': input_ids,
                  'labels': labels}

        return model(**inputs)  # can use ** with dict input


def classifier(features, model, labels, randomize=False):
    if randomize:
        return randomize_classifier(features, model, labels)


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

    num_choices = input_ids.shape[1]

    input_ids = input_ids.view(-1, input_ids.size(-1))
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

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(pretrained_model_name_or_path, config)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, **kwargs):

        outputs = self.bert(**kwargs)
        classification_scores = outputs

        classification_scores = classification_scores.view(-1, self.num_choices)

        if labels is not None:
            loss = self.BCEWithLogitsLoss(classification_scores, labels)
            return classification_scores, loss

        return classification_scores, None


class MyBertForMultipleChoice(nn.Module):
    def __init__(self, pretrained_model_name_or_path, config):
        super(MyBertForMultipleChoice, self).__init__()
        self.bert = BertForMultipleChoice.from_pretrained(pretrained_model_name_or_path, config=config)

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(pretrained_model_name_or_path, config)

    def forward(self, my_attention_mask, labels, **kwargs):
        outputs = self.bert(**kwargs)
        classification_scores = outputs

        classification_scores = classification_scores.view(-1, self.num_choices)

        if labels is not None:
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