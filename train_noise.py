import os, glob
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
import logging
import csv
from tqdm import trange, tqdm
import torch.optim as optim
import nltk
from nltk.corpus import stopwords
import getpass
from collections import Counter

#logging
logger = logging.getLogger(__name__)

# whether able to use gpu or not
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# don't want to have stop words in answers be labeled important so remove them when in answers
stop_words = set(stopwords.words('english'))

def set_seed(args):
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

# function to run through data provided by "Learning What is Essential in Questions" Khashabi et al. and produce
# a tokenized list of all question answer pairings, a label of each word 0 <= ell <= 1, input masks to say where
# non-padding tokens are, and a word to id dict to map words to their tokens
def get_words(cutoff):
    data_filename = '../ARC/essential_data/turkerSalientTermsWithOmnibus-v3.tsv'
    words_list = []
    labels_list = []

    # record number of entries lost to no answers
    num_lost = 0

    with open(data_filename, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file)

        for line in tsv_reader:
            # wasn't sure how to split by tabs so did it manually
            line = ''.join(line)
            line = line.split('\t')

            # separate text into the three categories
            question_text = line[0]
            num_annotators = int(line[1])
            annotations = line[2] # this is the raw number of annotators that believed it was an important word

            # from looking at data this is how all answers are indicated
            answer_letters = ['(A)', '(B)', '(C)', '(D)', '(1)', '(2)', '(3)', '(4)']

            # if no answers provided note and continue without adding to returned data
            if not any([answer_letter in question_text for answer_letter in answer_letters]):
                logger.info('One of {} is not in the question text: {}. Skipping.'.format(', '.join(answer_letters), question_text))
                num_lost += 1
                continue

            # get the actual answer indicators used
            answer_letters = [a_l for a_l in answer_letters if a_l in question_text]

            # where each answer begins/ends in question text, should have len(answer_letters) + 1 elements
            answer_inds = [question_text.index(answer_letter) for answer_letter in answer_letters] + [len(question_text)]

            # beginning and ending indice for each answer, should have len(answer_letters) elements
            answer_ind_tuples = [(answer_inds[i], answer_inds[j]) for i, j in zip(range(len(answer_inds)-1), range(1, len(answer_inds)))]

            answers = []
            answer_scores = []
            # record and store each answer, assign each answer word a 1 indicating all annotators would have thought it to be important
            for t in answer_ind_tuples:
                answer_words = question_text[t[0]:t[1]].rstrip().split(' ')
                answer_words = [aw.lower() for aw in answer_words[1:] if aw not in stop_words]

                answers.append(answer_words)
                answer_scores.append([1.]*len(answer_words))

            # get words and their scores
            annotations = annotations.split('|')
            words_and_scores = [(annotation[:-1].lower(), int(annotation[-1])) for annotation in annotations]
            assert all([score <= num_annotators for _, score in words_and_scores])

            # change from raw score to percentage of annotators
            words_and_percents = [(w, s/num_annotators) for w, s in words_and_scores]

            # split apart
            q_words, percents = map(list, zip(*words_and_percents))

            # add answer words and answer scores to end of question words and percents
            for answer, answer_score in zip(answers, answer_scores):

                words = ['[BOS]'] + q_words + answer + ['[EOS]']
                labels = [0.] + percents + answer_score + [0.]

                words_list.append(words)
                labels_list.append(labels)

            if cutoff is not None and len(words_list) >= cutoff:
                break

    logger.info('There are {} lines in the data'.format(len(words_list)))
    logger.info('Lost {} to no answers'.format(num_lost))

    padding_tokens = ['[EOS]', '[BOS]']
    token_list = []
    word_to_idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
    counter = None
    # create a dictionary to translate from words to tokens to be used in an embedding matrix
    # also edits labels, and words to fix typos in sentences, specifically two words sticking together, i.e. oceanbreeze -> ocean breeze
    for ind1, (sent, labels) in enumerate(zip(words_list, labels_list)):
        new_sent = []
        new_labels = []
        for ind2, (word, label) in enumerate(zip(sent, labels)):
            # remove non numbers and letters, unless it's a word used in padding
            word = ''.join([c for c in word if c.isalnum()]) if word not in padding_tokens else word

            # try to find and fix minor typos
            if not nlp.vocab.has_vector(word) and word not in padding_tokens:
                # create a list of proposed pairs of words by adding a space in between each letter
                spaced_tokens = [(word[:i], word[i:]) for i in range(1, len(word))]
                # see if both spaced words have a word vector
                temp = [all((nlp.vocab.has_vector(st[0]), nlp.vocab.has_vector(st[1]))) for st in spaced_tokens]

                # if any do then update labels and words to replace typo word with two new words that share the label of the typo word
                if any(temp):
                    if sum(temp) > 1:
                        logger.info('More than one acceptable pairing of words: {}'.format(' '.join(['({}, {})'.format(st[0], st[1]) for st, t in zip(spaced_tokens, temp) if t])))

                    # only look at first success, may not be true
                    best_words = spaced_tokens[temp.index(True)]

                    # add each word to translation dict and update new sentence and labels
                    for best_word in best_words:
                        if best_word not in word_to_idx:
                            word_to_idx[best_word] = len(word_to_idx)

                        new_sent.append(word_to_idx[best_word])
                    new_labels.extend([label]*2)

                    # continue to make sure the typo word is not added to the translation dict
                    continue

            # if the word had an embedding vector add it to the translation dict if not in already
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

            # update current sentence with token and label
            new_sent.append(word_to_idx[word])
            new_labels.append(label)

        assert len(new_sent) == len(new_labels)

        # add new tokenized sentence and edit labels in case a typo word was replaced
        token_list.append(new_sent)
        labels_list[ind1] = new_labels

        # update counter to be used when sampling words
        if counter is None:
            counter = Counter(sent)
        else:
            counter.update(sent)

    # find max length of tokenized sentence for padding purposes
    # need to pad so input is the same size for MLP in classification, not neccessary for LSTM though
    max_length = max([len(t) for t in token_list])

    all_input_masks = []
    all_tokens_list = []
    all_labels_list = []

    # pad token and labels up to max length
    # also create input masks to identify where padding is, EOS and BOS are not considered padding here, not sure if that's correct
    for tokens, labels in zip(token_list, labels_list):
        assert len(tokens) == len(labels), 'The length of tokens is {} and the length of labels is {}'.format(len(tokens), len(labels))
        padding_length = max_length - len(tokens)
        input_mask = [1]*len(tokens) + [0]*padding_length

        all_input_masks.append(input_mask)
        all_tokens_list.append(tokens + [0]*padding_length)
        all_labels_list.append(labels + [0.]*padding_length)

    assert all([len(input) == len(tokens) == len(labels) == max_length for input, tokens, labels in zip(all_input_masks, all_tokens_list, all_labels_list)])

    return all_tokens_list, all_input_masks, all_labels_list, word_to_idx, counter


# create an embedding matrix from GloVe vectors
def create_embedding_matrix(word_to_idx):
    # known a priori that each vector has dimension 300
    embedding_matrix = torch.empty((len(word_to_idx), 300))

    assert isinstance(word_to_idx, dict)
    for word, id in word_to_idx.items():
        # create an embedding vector in row of token if an embedding exists, otherwise put random noise there
        if nlp.vocab.has_vector(word):
            embedding_matrix[id, :] = torch.tensor(nlp.vocab.get_vector(word))

        else:
            logger.info('The token {} does not have a vector. Replacing with noise.'.format(word))
            embedding_matrix[id, :] = torch.rand((300,))

    return embedding_matrix


class LSTM2MLP(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(LSTM2MLP, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_matrix.shape[1]

        self.hidden_dim = args.essential_terms_hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.linear = nn.Sequential(nn.Linear(self.hidden_dim, 1),
                                    nn.Sigmoid())
        # using standard MSE loss
        self.loss = nn.MSELoss(reduction='none')

        self.device = args.device

    def forward(self, input_ids, labels=None, input_masks=None, add_special_tokens=False):
        # each should be batch_size x max_length

        batch_size = input_ids.shape[0]

        # if evaluating no labels provided so create a non-vector since we don't care about errors
        labels = labels if labels is not None else torch.zeros((batch_size, input_ids.shape[1]), dtype=torch.float)

        # when doing analysis add a BOS and EOS token and labels of zeros on each end
        if add_special_tokens:
            input_ids = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), input_ids, 2*torch.ones((batch_size, 1), dtype=torch.long)), dim=1)
            labels = torch.cat((torch.zeros((batch_size, 1), dtype=torch.float), labels, torch.zeros((batch_size, 1), dtype=torch.float)), dim=1)

        max_length = input_ids.shape[1]

        # when doing analysis initialize input mask to indicate no padding, expecting everything to come through one at a time
        if input_masks is None:
            input_masks = torch.ones((batch_size, max_length), dtype=torch.long)
            if add_special_tokens:
                input_masks[:, 0] = 0
                input_masks[:, -1] = 0
            input_masks = input_masks.to(self.device)

        labels = labels.to(self.device)

        # from tokens create embedded sentences from embedding matrix
        inputs = torch.empty((batch_size, max_length, self.embedding_dim)).to(self.device)
        for s_ind, sentence in enumerate(input_ids):
            for m_ind in range(max_length):
                inputs[s_ind, m_ind, :] = self.embedding_matrix[sentence[m_ind]]

        # get output of each input
        lstm_out, (last_hidden, last_cell) = self.lstm(inputs)
        lstm_out = torch.mean(lstm_out.view(batch_size, max_length, 2, self.hidden_dim), dim=2)

        # expecting lstm_out to be batch_size x max_length x hidden_dim
        out_scores = self.linear(lstm_out.view(-1, self.hidden_dim))

        # expecting out_scores to be batch_size*max_length x 1
        out_errors = self.loss(out_scores, labels.reshape((-1, 1)))

        # get rid of errors from the padding
        out_errors = out_errors * input_masks.reshape((-1, 1))

        # only count erros from the non-padded tokens
        sum_errors = torch.sum(out_errors.view((batch_size, max_length)), dim=1)
        sum_masks = torch.sum(input_masks, dim=1)
        out_errors = torch.mean(sum_errors/sum_masks)

        # reshape the scores to reflect the input dimension
        out_scores = out_scores.reshape((batch_size, max_length))

        # get rid of scores of BOS and EOS tokens
        if add_special_tokens:
            out_scores = out_scores[:, 1:-1]

        return out_scores, out_errors


# save the model parameters, embedding matrix, and translation dict
def save_model(model, embedding_matrix, hidden_dim, word_to_idx, counter):
    model_save_file = 'saved/train_noise/essential_terms_model_parameters_hidden{}.pt'.format(hidden_dim)

    with open(model_save_file, 'wb') as mf:
        torch.save(model.state_dict(), mf)

    embedding_matrix_save_file = 'saved/train_noise/embedding_matrix.py'

    with open(embedding_matrix_save_file, 'wb') as ef:
        torch.save(embedding_matrix, ef)

    word_to_idx_save_file = 'saved/train_noise/word_to_idx.py'

    with open(word_to_idx_save_file, 'wb') as wf:
        torch.save(word_to_idx, wf)

    counter_save_file = 'saved/train_noise/counter.py'

    with open(counter_save_file, 'wb') as cf:
        torch.save(counter, cf)


# to be used in analysis, load model and translation dict
def load_model(args):
    filenames = glob.glob('saved/train_noise/*hidden*.pt')

    model_filename = [f for f in filenames if f[(f.index('hidden')+len('hidden')):f.index('.pt')] == str(args.essential_terms_hidden_dim)]

    assert len(model_filename) == 1

    embedding_matrix_save_file = 'saved/train_noise/embedding_matrix.py'
    if os.path.exists(embedding_matrix_save_file):
        with open(embedding_matrix_save_file, 'rb') as ef:
            embedding_matrix = torch.load(ef)

    model = LSTM2MLP(embedding_matrix=embedding_matrix, args=args)
    model.load_state_dict(torch.load(model_filename[0], map_location=torch.device('cpu') if getpass.getuser() == 'Mitch' else None))
    model.eval()

    word_to_idx_save_file = 'saved/train_noise/word_to_idx.py'

    with open(word_to_idx_save_file, 'rb') as wf:
        word_to_idx = torch.load(wf)

    counter_save_file = 'saved/train_noise/counter.py'

    with open(counter_save_file, 'rb') as cf:
        counter = torch.load(cf)

    return model, word_to_idx, counter


# train and save model
def train(args):

    all_token_list, all_input_masks, all_labels_list, word_to_idx, counter = get_words(args.cutoff)

    embedding_matrix = create_embedding_matrix(word_to_idx)

    # throw embedding matrix to device
    embedding_matrix = embedding_matrix.to(args.device)

    # initialize model
    model = LSTM2MLP(embedding_matrix=embedding_matrix, args=args)

    # throw model to device
    model.to(args.device)

    # intiailize optimizer
    optimizer = optim.Adam(params=model.parameters())

    # create TorchTensorDatset
    tokens_tensor = torch.tensor(all_token_list, dtype=torch.long)
    input_masks_tensor = torch.tensor(all_input_masks, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels_list, dtype=torch.float)

    dataset = TensorDataset(tokens_tensor, input_masks_tensor, labels_tensor)

    # create sampler to process batches
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    # start training
    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))
        for iteration, batch in enumerate(epoch_iterator):

            # clear gradients in model
            model.zero_grad()

            # get batch
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'input_masks': batch[1],
                      'labels': batch[2],
                      }

            # send through model
            model.train()
            _, error = model(**inputs)

            # backwards pass
            error.backward()
            optimizer.step()

            # print('model params')
            # for i in range(len(list(model.parameters()))):
            #     print(i)
            #     print(list(model.parameters())[i].grad)
            #     print(torch.max(list(model.parameters())[i].grad))
            # print('hi')

            logger.info('The error is {}'.format(error))

    save_model(model, embedding_matrix, args.essential_terms_hidden_dim, word_to_idx, counter)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    if not getpass.getuser() == 'Mitch':

        # Required
        parser.add_argument('--epochs', default=None, type=int, required=True,
                            help='Number of epochs to train model')
        parser.add_argument('--batch_size', default=None, type=int, required=True,
                            help='Batch size of each iteration')
        parser.add_argument('--essential_terms_hidden_dim', default=None, type=int, required=True,
                            help='Dimension size of hidden layer')

        # Optional
        parser.add_argument('--cutoff', default=None, type=int,
                            help='Cutoff number of examples when testing')
        parser.add_argument('--seed', default=1234, type=int,
                            help='Seed for randomization')

        args = parser.parse_args()
    else:
        class Args(object):
            def __init__(self):
                self.epochs = 10
                self.batch_size = 12
                self.essential_terms_hidden_dim = 100

                self.cutoff = None
                self.seed = 1234
        args = Args()

    # Setup logging
    num_noise_logging_files = len(glob.glob('logging/loggingnoise_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/loggingnoise_{}'.format(num_noise_logging_files))

    #get device
    args.device = get_device()

    # output device
    logger.info('Using device: {}'.format(args.device))

    # set seed
    set_seed(args)

    # parser/ embeddings
    nlp = spacy.load("en_core_web_md", disable=['ner', 'parser'])

    train(args)