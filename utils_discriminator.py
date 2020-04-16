import torch
import torch.nn as nn
import logging
import os
import json

logger = logging.getLogger(__name__)

class DiscriminatorConfig(object):
    def __init__(self, **config_kwargs):
        for key, value in config_kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

class DiscriminatorLSTM(nn.Module):
    def __init__(self, config):
        super(DiscriminatorLSTM, self).__init__()

        self.config = config

        self.embedding_matrix_init()

        self.linear1 = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim, batch_first=True, bidirectional=False,
                            num_layers=config.num_layers, dropout=config.dropout)
        self.linear2 = nn.Sequential(nn.Linear(config.hidden_dim, 1),
                                     nn.Tanh())

    @staticmethod
    def Wloss(preds, labels, my_attention_mask):
        return -1 * torch.sum(preds * labels * my_attention_mask) / float(my_attention_mask.nonzero().shape[0])

    def embedding_matrix_init(self):
        assert hasattr(self, "config")

        if self.config.embedding_type is None:
            self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim, padding_idx=0)
        else:
            raise NotImplementedError

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        model_to_save = self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'linear_weights.pt')
        torch.save(model_to_save.state_dict(), output_model_file)

        assert hasattr(model_to_save, "config")
        config_filename = os.path.join(save_directory, 'config.json')
        with open(config_filename, 'w') as cf:
            json.dump(vars(model_to_save.config), cf)

        logger.info("Model weights and config saved in {}".format(output_model_file))


    @classmethod
    def from_pretrained(cls, config, pretrained_model_path):
        if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
            config_filename = os.path.join(pretrained_model_path, 'config.json')
            with open(config_filename, 'r') as cf:
                config_json = json.load(cf)
            config = DiscriminatorConfig(**config_json)

            model_to_return = cls(config)
            model_load_filename = os.path.join(pretrained_model_path, 'linear_weights.pt')
            model_to_return.load_state_dict(torch.load(model_load_filename))

            return model_to_return

        return cls(config)

    def forward(self, input_ids, discriminator_labels, my_attention_mask, **kwargs):

        temp_input_ids = input_ids.view(-1, input_ids.shape[-1])

        if 'inputs_embeds' in kwargs:

            inputs_embeds = kwargs['inputs_embeds']
            temp_inputs_embeds = torch.mm(inputs_embeds, self.embedding.weight)

            word_embeddings = temp_inputs_embeds.view(*temp_input_ids.shape, -1)

        else:
            word_embeddings = self.embedding(temp_input_ids)

        transformed_word_embeddings = self.linear1(word_embeddings)

        lstm_out, (last_hidden, last_cell) = self.lstm(transformed_word_embeddings)
        if self.lstm.bidirectional:
            lstm_out = torch.mean(lstm_out.view(transformed_word_embeddings.shape[0], transformed_word_embeddings.shape[1],
                                            2, -1), dim=2)

        logits = self.linear2(lstm_out)
        logits = logits.view(*discriminator_labels.shape)

        loss = self.Wloss(logits, discriminator_labels, my_attention_mask)

        return logits, loss


discriminator_models_and_config_classes = {
    'lstm': (DiscriminatorConfig, DiscriminatorLSTM),
}





