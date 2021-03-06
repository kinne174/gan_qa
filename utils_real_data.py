import os, glob
from random import shuffle
import json_lines, json
import tqdm
import logging
import codecs
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import getpass
from collections import Counter
from string import punctuation
import csv

logger = logging.getLogger(__name__)

stop_words = set(stopwords.words('english'))

class InputExample(object):

    def __init__(self, example_id, question, contexts, endings, classification_label, sentences_type):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.classification_label = classification_label
        self.sentences_type = sentences_type


def example_loader(args, subset):
    # returns an object of type ArcExample similar to hugging face transformers

    if args.task_name == 'arc':
        return arc_example_loader(args, subset)

    if args.task_name == 'race':
        return race_example_loader(args, subset)

    if args.task_name == 'swag':
        return swag_example_loader(args, subset)

def race_example_loader(args, subset):
    all_examples = []

    high = os.path.join(args.data_dir, "RACE/{}/high".format(subset))
    middle = os.path.join(args.data_dir, "RACE/{}/middle".format(subset))

    race_dirs = [high, middle]
    if args.task_difficulty == 'high':
        race_dirs = [high]
    elif args.task_difficulty == 'middle':
        race_dirs = [middle]

    for race_dir in race_dirs:
        files = glob.glob(race_dir + '/*txt')
        for file in tqdm.tqdm(files, desc='Creating RACE data from {}'.format(race_dir)):
            with open(file, 'r', encoding='utf-8') as handle:
                data_raw = json.load(handle)
                data_raw['race_id'] = '{}-{}'.format(subset, file)

            article = data_raw['article']
            for i in range(len(data_raw['answers'])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                all_examples.append(
                    InputExample(
                        example_id=data_raw['race_id'],
                        question=question,
                        contexts=[article, article, article, article],
                        endings=[options[0], options[1], options[2], options[3]],
                        classification_label=truth,
                        sentences_type=1
                    )
                )

    return all_examples


def swag_example_loader(args, subset):
    subset = 'val' if subset == 'dev' else subset

    data_filename = os.path.join(args.data_dir, 'SWAG/{}.csv'.format(subset))

    with open(data_filename, 'r', encoding='utf-8') as handle:
        lines = list(csv.reader(handle))

    if subset == 'train' and not lines[0][-1] == 'label':
        raise Exception('Input file must contain label column for training!')

    all_examples = []
    for line in lines[1:]:
        all_examples.append(
            InputExample(
                example_id=line[2],
                question=line[5],
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                classification_label=line[11],
                sentences_type=1
            )
        )

    return all_examples

def arc_example_loader(args, subset):
    # bad ids, each has at least one answer that does not contain any context
    # if another question answer task is used this will need to be fixed
    bad_ids = ['OBQA_9-737', 'OBQA_45', 'OBQA_750', 'OBQA_7-423', 'OBQA_619', 'OBQA_9-778', 'OBQA_10-201', 'OBQA_10-791', 'OBQA_10-1138', 'OBQA_12-717', 'OBQA_13-129', 'OBQA_13-468', 'OBQA_13-957', 'OBQA_14-10', 'OBQA_14-949', 'OBQA_14-1140', 'OBQA_14-1274']

    difficulty_id = None
    if args.task_difficulty == 'challenge':
        difficulty_id = 'ARCCH'
    elif args.task_difficulty == 'easy':
        difficulty_id = 'ARCEZ'



    all_examples = []
    data_filename = os.path.join(args.data_dir, '{}.jsonl'.format(subset))

    if args.use_corpus and subset is 'train':
        counter = Counter()

    # TODO add in one hop questions by going through twice and saving added ids, creating next best words to look for with counter of common co occurances that are not stop words

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

            # find if any of the domain words are in the tokenization of the question
            question_words = word_tokenize(question_text.lower())
            question_in_domain = any([dw in question_words for dw in args.domain_words])

            if not question_in_domain:
                # if the question has a domain word automatically save it, otherwise check the answers too
                for answer in answer_texts:
                    answer_words = word_tokenize(answer.lower())
                    question_in_domain = any([dw in answer_words for dw in args.domain_words])

                    if question_in_domain:
                        break

            if not question_in_domain and len(args.domain_words):
                continue

            if difficulty_id is not None and difficulty_id not in id:
                continue

            # update list of examples
            all_examples.append(InputExample(example_id=id,
                                           question=question_text,
                                           contexts=contexts,
                                           endings=answer_texts,
                                           classification_label=label,
                                           sentences_type=1))

            if args.use_corpus and subset is 'train':
                answer_words = word_tokenize(' '.join([at.lower() for at in answer_texts]))
                all_words = question_words + answer_words
                combined_words = set([w for w in all_words if w not in stop_words and not any([c in punctuation for c in w])])
                counter.update(combined_words)

            if args.cutoff is not None and len(all_examples) >= args.cutoff and subset == 'train':
                break

    num_examples = len(all_examples)
    logger.info('The number of questions found with domain words "{}" is {}'.format(' '.join(args.domain_words), num_examples))

    if args.use_corpus and subset is 'train':
        top_words = counter.most_common(10+len(args.domain_words))
        keywords_list = [t[0] for t in top_words if t[0] not in args.domain_words]
        keywords_dict = {kw: [] for kw in keywords_list}
        logger_ind = len(all_examples) // 1000

        data_filename = '../ARC/ARC-V1-Feb2018-2/ARC_Corpus.txt'
        with codecs.open(data_filename, 'r', encoding='utf-8', errors='ignore') as corpus:

            all_valid_sentences = []
            sentence_ind = 0
            # this will show up when running on console
            for line_ind, line in tqdm.tqdm(enumerate(corpus), desc='Searching corpus for examples.', mininterval=1):

                # tokenize and remove words that contain non alphanumeric characters
                sentence_words = word_tokenize(line.lower())
                sentence_words = [w for w in sentence_words if w.isalnum()]
                sentence = ' '.join(sentence_words)

                # determine if the sentence has any domain words
                keywords_in_sentence = [kw in sentence_words for kw in keywords_list]
                if not any(keywords_in_sentence):
                    continue
                # continue if sentence is too short
                elif len(sentence_words) <= 5:
                    continue
                else:
                    # save which sentence had the keyword in it, multiple keywords per sentence is allowed
                    for kw_ind, kw_bool in enumerate(keywords_in_sentence):
                        if kw_bool:
                            keywords_dict[keywords_list[kw_ind]].append(sentence_ind)

                    all_valid_sentences.append(sentence)
                    sentence_ind += 1

                if min([len(val) for val in keywords_dict.values()]) >= 2 and getpass.getuser() == 'Mitch':
                    break

                if line_ind >= 5000000:
                    break

        # sentences should be in groups of 7-10, try to maximize how many left over sentences there are (should be atleast 6 or whatevers greatest)
        num_sentences = {10: None, 9: None, 8: None, 7: None}
        num_sentences_to_keep = None
        for kw, sinds in keywords_dict.items():
            for num in num_sentences.keys():
                # try in order from 10 - 7. If an acceptable number of leftover sentences is found keep those
                num_mod = len(sinds) % num
                if num_mod >= 6:
                    num_sentences_to_keep = num
                    break
                num_sentences[num] = num_mod
            # if an acceptable mod isn't found use the grouping number with highest mod
            if num_sentences_to_keep is None:
                num_sentences_to_keep = max(num_sentences.items(), key= lambda t: t[1])[0]

            # not a lot of reason to shuffle here, but also not a lot of reason not too... so why not!
            shuffle(sinds)

            # should be all contexts so this is just generic to keep form with questions format above
            question_text = ''
            answer_texts = [''] * 4
            for i in range((len(sinds)//num_sentences_to_keep) + 1):  # +1 to make sure the leftover sentences are included
                contexts = []
                partitioned_sinds = sinds[i * num_sentences_to_keep:(i + 1) * num_sentences_to_keep]
                for _ in range(4):
                    # shuffle here to randomize order to give generator more practice seeing sentences
                    shuffle(partitioned_sinds)
                    contexts.append(' '.join([all_valid_sentences[sind] for sind in partitioned_sinds]))


                # update list of examples
                all_examples.append(InputExample(example_id='{}-{}'.format(kw, i),
                                               question=question_text,
                                               contexts=contexts,
                                               endings=answer_texts,
                                               sentences_type=0,
                                               classification_label=0))  # classification_label has be to be something but should be disregarded with sentence_type 0s

                # informative
                if len(all_examples) >= logger_ind * 1000:
                    logger.info('Writing {}th example.'.format(logger_ind * 1000))
                    logger_ind += 1


            logger.info('The number of examples added with one-hop word "{}" is {}'.format(kw, len(all_examples)-num_examples))
            num_examples = len(all_examples)

            # mainly for code-testing purposes
            if args.cutoff is not None and len(all_examples) >= args.cutoff*2 and subset == 'train':
                break


    # make sure there is at least one example
    assert len(all_examples) > 1

    # informative
    logger.info('Number of examples in {} subset is {}'.format(subset, len(all_examples)))

    return all_examples
