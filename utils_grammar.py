import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger(__name__)

def create_dataset(args, tokenizer):
    raise NotImplementedError


def load_and_cache_dataset(args, tokenizer):
    dataset_filename = ''
    if os.path.exists(dataset_filename):
        dataset = torch.load(dataset_filename)
    else:
        dataset = create_dataset(args, tokenizer)
        torch.save(dataset, dataset_filename)

    return dataset


class GrammarLSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, cls_id, sep_id, device):
        super(GrammarLSTM, self).__init__()

        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        self.hidden_dim = hidden_dim

        self.LSTM = nn.LSTM(embedding_dim, 1, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.Linear = nn.Linear(hidden_dim, 1)
        self.Loss = nn.BCEWithLogitsLoss()

        self.cls_id = cls_id
        self.sep_id = sep_id

    def forward(self, input_ids, labels=None, add_special_tokens=False):

        batch_size = input_ids.shape[0]

        # when doing analysis add a CLS and SEP id on each end
        if add_special_tokens:
            input_ids = torch.cat((self.cls_id * torch.ones((batch_size, 1), dtype=torch.long), input_ids,
                                   self.sep_id * torch.ones((batch_size, 1), dtype=torch.long)), dim=1)

        # from ids create embedded sentences from embedding matrix
        inputs = self.embedding(input_ids)

        # get output of each input
        lstm_out, (last_hidden, last_cell) = self.LSTM(inputs)
        lstm_out = torch.mean(lstm_out.view(batch_size, 1, 2, self.hidden_dim), dim=2)
        lstm_out = lstm_out.squeeze()

        out_scores = self.Linear(lstm_out)

        # expecting out_scores to be batch_size x 1
        out_errors = self.Loss(out_scores, labels)

        # get rid of scores of BOS and EOS tokens
        if add_special_tokens:
            out_scores = out_scores[:, 1:-1]

        return out_scores, out_errors


def train(args, tokenizer):

    logging.info('Starting to train grammar model')

    dataset = load_and_cache_dataset(args, tokenizer)

    # initialize model
    vocabulary_size = tokenizer.vocabulary_size
    embedding_dim = args.grammar_embedding_dim
    hidden_dim = args.grammar_hidden_dim
    sep_id = tokenizer.sep_id
    cls_id = tokenizer.cls_id
    device = args.device

    model = GrammarLSTM(vocabulary_size, embedding_dim, hidden_dim, cls_id, sep_id, device)

    # throw model to device
    model.to(args.device)

    # intiailize optimizer
    optimizer = optim.Adam(params=model.parameters())

    # create sampler to process batches
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    # start training
    logger.info('Starting to train grammar model!')
    logger.info('There are {} examples.'.format(len(dataset)))

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))
        for iteration, batch in enumerate(epoch_iterator):

            # clear gradients in model
            model.zero_grad()

            # get batch
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'labels': batch[1],
                      }

            # send through model
            model.train()
            _, error = model(**inputs, add_special_tokens=True)

            # backwards pass
            error.backward()
            optimizer.step()

            logger.info('The error is {}'.format(error))

    save_model(args, model)

def save_model(args, model):
    raise NotImplementedError

def load_model(args):
    raise NotImplementedError
