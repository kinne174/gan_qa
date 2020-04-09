from utils_real_data import ArcExample
import torch
import tqdm
import logging
import nltk
from nltk import wordpunct_tokenize
from string import punctuation
import numpy as np

from train_noise import load_model

logger = logging.getLogger(__name__)


class ArcFeature(object):
    def __init__(self, example_id, choices_features, sentences_type, classification_label=None):
        # label is 0,1,2,3 depending on correct answer
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'token_type_mask': token_type_mask,
            'attention_mask': attention_mask
        } for input_ids, input_mask, token_type_mask, attention_mask in choices_features]
        self.sentences_type = sentences_type
        self.classification_label = classification_label
        self.discriminator_labels = [1]*4

class HuggingfaceTranslators:
    def __init__(self, transformer_name):
        valid_transformers = ['albert', 'roberta']
        assert transformer_name in valid_transformers, 'transformer must be one of {}'.format(' '.join(valid_transformers))

        self.transformer_name = transformer_name

    def AlbertTranslator(self, tokenizer, context_tokens, predictions):

        new_predictions = []
        shared_tokens = []
        prediction_ind = 0
        # if token is a special token then assign it zero (other than <unk>)
        for token in context_tokens:
            try:
                if token in tokenizer.all_special_tokens:
                    if token == '<unk>':
                        new_predictions.append(predictions[prediction_ind])
                        prediction_ind += 1
                    else:
                        new_predictions.append(0.)
                    shared_tokens.append(0)
                elif token in punctuation or token == '▁':
                    new_predictions.append(0.)
                    prediction_ind += 1
                    shared_tokens.append(0)
                elif '▁' not in token:  # don't know what this character is, had to copy paste from debugger
                    new_predictions.append(predictions[prediction_ind - 1])
                    shared_tokens.append(prediction_ind + 1)
                else:
                    new_predictions.append(predictions[prediction_ind])
                    shared_tokens.append(prediction_ind + 1)
                    prediction_ind += 1

            # This stuff should not activate but just in case...**
            except IndexError:
                break

        if len(new_predictions) > len(context_tokens):
            new_predictions = new_predictions[:len(context_tokens)]
            shared_tokens = shared_tokens[:len(context_tokens)]
        if len(new_predictions) < len(context_tokens):
            new_predictions.extend([np.mean(predictions)] * (len(context_tokens) - len(new_predictions)))
            shared_tokens.extend([shared_tokens[-1]] * (len(context_tokens) - len(shared_tokens)))
        # **

        return new_predictions, shared_tokens

    def RobertaTranslator(self, tokenizer, context_tokens, predictions):
        new_predictions = []
        shared_tokens = []
        prediction_ind = 0
        # if token is a special token then assign it zero (other than <unk>)
        for token_ind, token in enumerate(context_tokens):
            try:
                if token in tokenizer.all_special_tokens:
                    if token == tokenizer.unk_token:
                        new_predictions.append(predictions[prediction_ind])
                        prediction_ind += 1
                    else:
                        new_predictions.append(0.)
                    shared_tokens.append(0)
                elif 'Ġ' not in token and token_ind is not 0:  # don't know what this character is, had to copy paste from debugger
                    new_predictions.append(predictions[prediction_ind - 1])
                    shared_tokens.append(prediction_ind + 1)
                else:
                    new_predictions.append(predictions[prediction_ind])
                    shared_tokens.append(prediction_ind + 1)
                    prediction_ind += 1

            # This stuff should not activate but just in case...**
            except IndexError:
                break

        if len(new_predictions) > len(context_tokens):
            new_predictions = new_predictions[:len(context_tokens)]
            shared_tokens = shared_tokens[:len(context_tokens)]
        if len(new_predictions) < len(context_tokens):
            new_predictions.extend([np.mean(predictions)] * (len(context_tokens) - len(new_predictions)))
            shared_tokens.extend([shared_tokens[-1]] * (len(context_tokens) - len(shared_tokens)))
        # **

        return new_predictions, shared_tokens

    def translate(self, tokenizer, context_tokens, predictions):
        if self.transformer_name == 'albert':
            return self.AlbertTranslator(tokenizer, context_tokens, predictions)
        elif self.transformer_name == 'roberta':
            return self.RobertaTranslator(tokenizer, context_tokens, predictions)
        else:
            raise NotImplementedError


def feature_loader(args, tokenizer, examples):

    Translator = HuggingfaceTranslators(transformer_name=args.transformer_name)

    # load the model and translation dict and counter
    model, word_to_idx, _ = load_model(args)

    model = model.to(args.device)

    # returns a list of objects of type ArcFeature similar to hugging face transformers
    break_flag = False
    all_features = []
    for ex_ind, ex in tqdm.tqdm(enumerate(examples), desc='Examples to Features, num Examples: {}'.format(len(examples))):
        if ex_ind % 1000 == 0:
            logger.info('Converting example number {} of {} to features.'.format(ex_ind, len(examples)))
        assert isinstance(ex, ArcExample)

        choices_features = []
        for ending, context in zip(ex.endings, ex.contexts):
            question_ending = ex.question + ' ' + ending
            question_ending = ''.join([c if c.isalnum() else ' ' for c in question_ending])
            context = ''.join([c if c.isalnum() else ' ' for c in context])

            try:
                inputs = tokenizer.encode_plus(
                    question_ending,
                    context,
                    add_special_tokens=True,
                    truncation_strategy='only_second',
                    max_length=args.max_length,
                    return_token_type_ids=True,
                    return_special_tokens_mask=True,
                )
            except AssertionError as err_msg:
                logger.info('Assertion error at example id {}: {}'.format(ex_ind, err_msg))
                break_flag = True
                break

            if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                logger.info('Truncating context for question id {}'.format(ex.example_id))

            if args.transformer_name == 'albert':
                input_ids, token_type_mask = inputs['input_ids'], inputs['token_type_ids']
                # convert context ids to tokens
                context_beginning_ind = token_type_mask.index(1)
            elif args.transformer_name == 'roberta':
                input_ids, token_type_mask = inputs['input_ids'], inputs['special_tokens_mask']
                context_beginning_ind = token_type_mask[1:].index(1) + 1
            else:
                raise NotImplementedError

            input_mask = [1]*len(input_ids)

            words = wordpunct_tokenize(context.lower())
            # words = [''.join([c for c in word if c.isalnum()]) for word in words if word not in punctuation]

            # the model has it's own tokenizing so if the word is not there have to use the noise/ pad embedding
            tokens = []
            for word_ind, word in enumerate(words):
                if word in word_to_idx:
                    tokens.append(word_to_idx[word])
                else:
                    tokens.append(0)

            # send the sentence through the model
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

            # predictions is a vector the length of the sentence with a number between 0, 1 for each word representing how important the model believes it is, closer to 1 is more important
            predictions, _ = model(input_ids=tokens, add_special_tokens=True)

            assert tokens.shape == predictions.shape

            predictions = predictions.squeeze().tolist()

            # throw anything below 50th quantile to zero
            # quant = .5
            # quantile_ = np.quantile([p for p in predictions if p > 1e-6], quant)
            # predictions = [p if p >= quantile_ else 0. for p in predictions]

            # assert sum(predictions > quantile_)/len(predictions) >= 1-quant, 'The sum ({}) is not more than {}'.format(sum(predictions > quantile_)/len(predictions), 1-quant)

            # line up predictions with the context words

            context_ids = input_ids[context_beginning_ind:]
            context_tokens = tokenizer.convert_ids_to_tokens(context_ids)

            # for roberta disregard the G` alone and mark shared when there is no G` preceding the word
            new_predictions, shared_tokens = Translator.translate(tokenizer, context_tokens, predictions)

            if args.transformer_name == 'albert':
                assert len(new_predictions) == len(shared_tokens) == sum(token_type_mask), 'There should be the same number of predictions ({}) as shared_tokens ({}) as there are context tokens ({})'.format(len(new_predictions), len(shared_tokens), sum(token_type_mask))
            elif args.transformer_name == 'roberta':
                assert len(new_predictions) == len(shared_tokens), 'There should be the same number of predictions ({}) as shared_tokens ({})'.format(len(new_predictions), len(shared_tokens))

            attention_mask1 = [0]*context_beginning_ind + new_predictions
            attention_mask2 = [0]*context_beginning_ind + shared_tokens

            assert len(attention_mask1) == len(attention_mask2) == len(input_ids), 'The length of attention_masks ({}) ({}) should be the same length as input_ids ({})'.format(len(attention_mask1), len(attention_mask2), len(input_ids))

            padding_length = args.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [0]*padding_length
                token_type_mask = token_type_mask + [0]*padding_length
                input_mask = input_mask + [0]*padding_length
                attention_mask1 = attention_mask1 + [0.]*padding_length
                attention_mask2 = attention_mask2 + [0]*padding_length

            # concatenate lists
            attention_mask = attention_mask1 + attention_mask2

            assert len(input_ids) == args.max_length
            assert len(token_type_mask) == args.max_length
            assert len(input_mask) == args.max_length
            assert len(attention_mask) == 2*args.max_length

            choices_features.append((input_ids, input_mask, token_type_mask, attention_mask))

        if break_flag:
            break_flag = False
            continue

        if ex_ind == 0:
            logger.info('Instance of a Feature.\n input_ids are the transformations to the integers that the model understands\n'
                        'input_mask is 1 if there is a real word there and 0 for padding. \n'
                        'token_type_mask is 0 for the first sentence (question answer text) and 1 for the second sentence (context) and 0 for padding *this is odd* \n'
                        'attention_mask is 0 for question answer text and 1 for context and 0 for padding')
            logger.info('Question ID: {}'.format(ex.example_id))
            logger.info('input_ids: {}'.format(' '.join(map(str, input_ids))))
            logger.info('input_mask: {}'.format(' '.join(map(str, input_mask))))
            logger.info('token_type_mask: {}'.format(' '.join(map(str, token_type_mask))))
            logger.info('attention_mask: {}'.format(' '.join(map(str, attention_mask))))

        all_features.append(ArcFeature(example_id=ex.example_id,
                                       choices_features=choices_features,
                                       sentences_type=ex.sentences_type,
                                       classification_label=ex.classification_label,))

    return all_features


def load_features(cache_filename):
    logger.info('Loading features from {}'.format(cache_filename))
    features = torch.load(cache_filename)

    return features


def save_features(features, cache_filename):
    logger.info('Saving features into {}'.format(cache_filename))
    torch.save(features, cache_filename)

    return -1
