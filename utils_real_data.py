import os
import numpy as np
import string
import random
import json_lines
import tqdm
import logging

logger = logging.getLogger(__name__)


class ArcExample(object):

    def __init__(self, example_id, question, contexts, endings, label):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


def randomize_example_loader(cutoff):
    all_examples = []
    max_sentence_length = 15
    num_examples = 16 if cutoff is None else cutoff
    for _ in range(num_examples):
        question = ''.join(random.choice(string.ascii_letters) for _ in range(np.random.randint(max_sentence_length//2, max_sentence_length)))
        question += ' ' * (max_sentence_length - len(question))

        contexts = [
            ''.join(random.choice(string.ascii_letters) for _ in range(np.random.randint(max_sentence_length*3//2, max_sentence_length * 3))) for
            _ in range(4)]
        contexts = [c + ' ' * (max_sentence_length * 3 - len(c)) for c in contexts]

        endings = [''.join(random.choice(string.ascii_letters) for _ in range(np.random.randint(max_sentence_length//2, max_sentence_length)))
                   for _ in range(4)]
        endings = [e + ' ' * (max_sentence_length - len(e)) for e in endings]

        label = np.random.randint(4)

        all_examples.append(
            ArcExample(example_id=np.random.randint(num_examples),
                       question=question,
                       contexts=contexts,
                       endings=endings,
                       label=label
                       )
        )

    return all_examples


def example_loader(args, subset):
    # returns an object of type ArcExample similar to hugging face transformers

    # bad ids, each has at least one answer that does not contain any context
    # if another question answer task is used this will need to be fixed
    bad_ids = ['OBQA_9-737', 'OBQA_45', 'OBQA_750', 'OBQA_7-423', 'OBQA_619', 'OBQA_9-778', 'OBQA_10-201', 'OBQA_10-791', 'OBQA_10-1138', 'OBQA_12-717', 'OBQA_13-129', 'OBQA_13-468', 'OBQA_13-957', 'OBQA_14-10', 'OBQA_14-949', 'OBQA_14-1140', 'OBQA_14-1274']

    all_examples = []
    data_filename = os.path.join(args.data_dir, '{}.jsonl'.format(subset))
    with open(data_filename, 'r') as file:
        jsonl_reader = json_lines.reader(file)

        # this will show up when running on console
        for ind, line in tqdm.tqdm(enumerate(jsonl_reader), desc='Creating {} examples.'.format(subset), mininterval=1):
            if ind % 1000 == 0:
                logger.info('Writing example number {}'.format(ind))

            id = line['id']

            if id in bad_ids:
                continue

            # if the number of options is not equal to 4 update the logger and skip it, all of the formatting works with
            # 4 options, maybe can update in the future to put a dummy one there or set the probability to 0 that it is
            # selected as correct later
            if len(line['question']['choices']) != 4:
                logger.info('Question id {} did not contain four options. Skipped it.'.format(id))
                continue

            label = line['answerKey']
            if label not in '1234':
                if label not in 'ABCD':
                    logger.info('Question id {} had an incorrect label of {}. Skipped it'.format(id, label))
                    continue
            # label should be the position in the list that the correct answer is
                else:
                    label = ord(label) - ord('A')
            else:
                label = int(label) - 1

            # extract question text, answer texts and contexts
            question_text = line['question']['stem']
            contexts = [c['para'] for c in line['question']['choices']]
            answer_texts = [c['text'] for c in line['question']['choices']]

            # update list of examples
            all_examples.append(ArcExample(example_id=id,
                                           question=question_text,
                                           contexts=contexts,
                                           endings=answer_texts,
                                           label=label))

            if args.cutoff is not None and len(all_examples) >= args.cutoff and subset == 'train':
                break

    # make sure there is at least one example
    assert len(all_examples) > 1

    # informative
    logger.info('Number of examples in {} subset is {}'.format(subset, len(all_examples)))

    return all_examples
