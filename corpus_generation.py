import os
import codecs
import json_lines
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import getpass
from string import punctuation
import tqdm
from collections import Counter
import bisect
import argparse

stop_words = set(stopwords.words('english'))

def BinarySearch(ell, x):
    j = bisect.bisect_left(ell, x)
    if j != len(ell) and ell[j] == x:
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser()

    if not getpass.getuser() == 'Mitch':

        # Required
        parser.add_argument('--domain_words', default=None, nargs='+', required=True,
                            help='Domain words to search for')
        parser.add_argument('--output_dir', default=None, type=str, required=True,
                            help='Folder where output should be sent')

        # Optional
        parser.add_argument('--corpus_cutoff', default=None, type=int,
                            help='Cutoff number of lines in corpus')
        parser.add_argument('--output_cutoff', default=None, type=int,
                            help='Cutoff number of examples when testing')
        parser.add_argument('--overwrite_output_file', action='store_true',
                            help='Overwrite the output directory')


        args = parser.parse_args()
    else:
        class Args(object):
            def __init__(self):
                self.domain_words = ['moon', 'earth']
                self.output_dir = 'output/temp/'

                self.corpus_cutoff = 50
                self.output_cutoff = None
                self.overwrite_output_file = True
        args = Args()

    common_words = Counter()
    all_valid_inds = []
    num_all_sentences = 0

    output_filename = os.path.join(args.output_dir, 'lm_sentences_{}.txt'.format('-'.join(args.domain_words)))
    if os.path.exists(output_filename) and not args.overwrite_output_file:
        raise Exception('The filename already exists and not being told to overwrite! {}'.format(output_filename))

    for i in range(2):
        if i == 0:
            keywords_list = args.domain_words
            write_flag = 'w'
        elif i == 1:
            most_common = common_words.most_common(7)
            keywords_list = [t[0] for t in most_common if t[0] not in args.domain_words]
            print('The most common words to search on are {}'.format(', '.join(keywords_list)))
            write_flag = 'a'
        else:
            raise NotImplementedError

        data_filename = '../ARC/ARC-V1-Feb2018-2/ARC_Corpus.txt'

        with codecs.open(data_filename, 'r', encoding='utf-8', errors='ignore') as corpus:

            all_valid_sentences = []

            # this will show up when running on console
            for line_ind, line in tqdm.tqdm(enumerate(corpus), desc='Searching corpus for examples step {}.'.format(i+1), mininterval=1):
                if i == 1 and BinarySearch(all_valid_inds, line_ind):
                    continue

                # tokenize and remove words that contain non alphanumeric characters
                line = line.lower()
                sentence_words = word_tokenize(line)
                sentence_words = [w for w in sentence_words if w.isalnum()]

                # determine if the sentence has any domain words
                keywords_in_sentence = [kw in sentence_words for kw in keywords_list]
                if not any(keywords_in_sentence):
                    continue
                # continue if sentence is too short
                elif len(sentence_words) <= 5:
                    continue
                else:
                    all_valid_sentences.append(line)

                    if i == 0:
                        common_words.update([w for w in sentence_words if w not in stop_words])
                        all_valid_inds.append(line_ind)

                    if len(all_valid_sentences) % 1000 == 0:
                        print('The number of sentences in step {} is {}'.format(i+1, len(all_valid_sentences)))

                    if args.output_cutoff is not None and len(all_valid_sentences) >= args.output_cutoff:
                        break

                if args.corpus_cutoff is not None and line_ind >= args.corpus_cutoff:
                    break


        with codecs.open(output_filename, write_flag) as handle:
            for sentence in all_valid_sentences:
                handle.write(sentence)

        num_all_sentences += len(all_valid_sentences)

    print('The total sentences is {}'.format(num_all_sentences))


if __name__ == '__main__':
    main()