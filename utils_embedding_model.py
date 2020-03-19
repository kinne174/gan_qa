import numpy as np
import getpass
from utils_real_data import ArcExample
import torch
import tqdm
import logging
import os

from train_noise import load_model

logger = logging.getLogger(__name__)


class ArcFeature(object):
    def __init__(self, example_id, choices_features, label=None):
        # label is 0,1,2,3 depending on correct answer
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'token_type_mask': token_type_mask,
            'attention_mask': attention_mask
        } for input_ids, input_mask, token_type_mask, attention_mask in choices_features]
        self.label = label


def feature_loader(args, tokenizer, examples):
    # returns a list of objects of type ArcFeature similar to hugging face transformers
    break_flag = False
    all_features = []
    for ex_ind, ex in tqdm.tqdm(enumerate(examples), desc='Examples to Features'):
        if ex_ind % 1000 == 0:
            logger.info('Converting example number {} of {} to features.'.format(ex_ind, len(examples)))
        assert isinstance(ex, ArcExample)

        choices_features = []
        for ending, context in zip(ex.endings, ex.contexts):
            question_ending = ex.question + ' ' + ending

            try:
                inputs = tokenizer.encode_plus(
                    question_ending,
                    context,
                    add_special_tokens=True,
                    truncation_strategy='only_second',
                    max_length=args.max_length
                )
            except AssertionError as err_msg:
                logger.info('Assertion error at example id {}: {}'.format(ex_ind, err_msg))
                break_flag = True
                break

            if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                logger.info('Truncating context for question id {}'.format(ex.example_id))

            input_ids, token_type_mask = inputs['input_ids'], inputs['token_type_ids']

            # TODO get the attention mask one time here and assign a value to each word, then attentionM can change them to zeros and ones to classify which words should be changed

            input_mask = [1]*len(input_ids)

            padding_length = args.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [0]*padding_length
                token_type_mask = token_type_mask + [0]*padding_length
                input_mask = input_mask + [0]*padding_length

            assert len(input_ids) == args.max_length
            assert len(token_type_mask) == args.max_length
            assert len(input_mask) == args.max_length

            # the token_type_mask and attention_mask is the same so can just use token_type_mask twice
            choices_features.append((input_ids, input_mask, token_type_mask, token_type_mask))

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
            logger.info('attention_mask: {}'.format(' '.join(map(str, token_type_mask))))

        all_features.append(ArcFeature(example_id=ex.example_id,
                                       choices_features=choices_features,
                                       label=ex.label))

    return all_features


def load_features(cache_filename):
    logger.info('Loading features from {}'.format(cache_filename))
    features = torch.load(cache_filename)

    return features


def save_features(features, cache_filename):
    logger.info('Saving features into {}'.format(cache_filename))
    torch.save(features, cache_filename)

    return -1
