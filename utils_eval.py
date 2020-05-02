import os, glob
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import nltk
from nltk import ngrams
from nltk.translate.bleu_score import corpus_bleu
from collections import Counter
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_gan as tfgan
import json
import matplotlib.pyplot as plt

from transformers import RobertaTokenizer

from utils_generator import generator_models_and_config_classes
from utils_attention import attention_models_and_config_classes
from utils_reinforce import word_index_selector
from utils_embedding_model import load_features

logger = logging.getLogger(__name__)

# returns a list of length num_choices with each entry a length four list with a 1 in the correct response spot and 0s elsewhere
def label_map(labels, num_choices):

    def label_list(target, new_target, switch_index, num_choices):
        l = [target]*num_choices
        l[switch_index] = new_target
        return l

    answers = [label_list(0, 1, lab, num_choices) for lab in labels]
    return answers

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

# evaluation for generator mainly, classification can be done with pre training on other models and comparing performance, possibly with other datasets or GLUE benchmarks
def main():

    class Args(object):
        def __init__(self):
            self.generator_directory = '/home/kinne174/private/PythonProjects/gan_qa/output/roberta-reinforce-roberta-reinforce-/saved/my_method/roberta-generator-120/'
            self.cache_dir = 'saved/'
            self.transformer_name = 'roberta'
            self.tokenizer_name = 'roberta-base'
            self.output_dir = 'output/'
            self.subset = 'dev'
            self.domain_words = []

            self.mu_p = 0.30
            self.perplexity_generator_directory = '/home/kinne174/private/Output/transformers_gpu/language_modeling/saved/force_roberta-base/'

            self.do_mode_collapse = True
            self.do_fid = False
            self.do_perplexity = True
            self.do_bleu = False

    args = Args()

    # Setup logging
    if not os.path.isdir('logging/generator_eval'):
        os.makedirs('logging/generator_eval')
    num_logging_files = len(glob.glob('logging/generator_eval/*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler('logging/generator_eval/eval-{}'.format(num_logging_files)),
                                  logging.StreamHandler()])

    args.output_dir = os.path.join(args.output_dir, 'generator_eval/')
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.cache_dir = os.path.join(args.cache_dir, args.transformer_name)
    args.device = 'cuda:5'
    logger.info('Device is {}'.format(args.device))

    # get tokenizer
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=True)

    logger.info('Loading Generator from {}'.format(args.generator_directory))
    _, generator_model_class = generator_models_and_config_classes['roberta-reinforce']
    # roberta, albert, seq, roberta-reinforce
    generator_model_dict = {'config': args.generator_directory,
                            'pretrained_model_name_or_path': args.generator_directory}
    generatorM = generator_model_class.from_pretrained(**generator_model_dict)

    attention_config_class, attention_model_class = attention_models_and_config_classes['essential-reinforce']
    attention_config_dict = {'mu_p': args.mu_p,
                             'mask_id': tokenizer.mask_token_id}
    attention_config = attention_config_class.from_pretrained(**attention_config_dict)
    attention_model_dict = {'config': attention_config}
    attentionM = attention_model_class.from_pretrained(**attention_model_dict)

    # push models to device
    attentionM.to(args.device)
    generatorM.to(args.device)

    all_saved_data = stream_data(args, generatorM, attentionM)

    discriminator_filename = os.path.join(args.output_dir, 'generator_eval-{}.txt'.format(num_logging_files))
    write_out_discriminator_eval(all_saved_data, tokenizer, discriminator_filename, args)


def load_and_cache_features(args, subset):
    assert subset in ['train', 'dev', 'test']

    cached_features_filename = os.path.join(args.cache_dir, '{}_{}_256_'.format(subset,
                                                                                args.tokenizer_name))
    if os.path.exists(cached_features_filename):
        logger.info('Loading features from ({})'.format(cached_features_filename))
        features = load_features(cached_features_filename)
    else:
        raise NotImplementedError

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_token_type_mask = torch.tensor(select_field(features, 'token_type_mask'), dtype=torch.long)
    all_attention_mask = torch.tensor(select_field(features, 'attention_mask'), dtype=torch.float)
    all_classification_labels = torch.tensor(label_map([f.classification_label for f in features], num_choices=4), dtype=torch.float)
    all_sentences_types = torch.tensor([f.sentences_type for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_mask, all_attention_mask, all_classification_labels, all_sentences_types)

    return dataset


def stream_data(args, generatorM, attentionM, subset, oracleM_dir):
    generatorM.eval()
    attentionM.eval()

    eval_dataset = load_and_cache_features(args, subset)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    all_saved_data = {}

    # num_of_grams = [2, 3, 4]

    # for n in num_of_grams:
    #     all_saved_data['fake {}-gram counter'.format(n)] = Counter()
    #     all_saved_data['real {}-gram counter'.format(n)] = Counter()

    _, generator_model_class = generator_models_and_config_classes['roberta-reinforce']
    # roberta, albert, seq, roberta-reinforce
    generator_model_dict = {'config': oracleM_dir,
                            'pretrained_model_name_or_path': oracleM_dir}
    oracleM = generator_model_class.from_pretrained(**generator_model_dict)
    oracleM.to(args.device)

    all_saved_data['all_perplexities'] = None

    all_saved_data['all_fake_input_ids'] = None

    with torch.no_grad():
        logger.info('Starting Evaluation!')
        for batch_ind, batch in tqdm(enumerate(eval_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'my_attention_mask': batch[3],
                      'classification_labels': batch[4],
                      'sentences_type': batch[5],
                      }

            masked_input_ids, my_attention_mask, _ = attentionM(inputs['input_ids'],
                                                                inputs['my_attention_mask'])
            inputs.update({'masked_input_ids': masked_input_ids,
                           'my_attention_mask': my_attention_mask
                           })

            fake_vocabulary_logprobs = generatorM(masked_input_ids, inputs['attention_mask'],
                                               inputs['token_type_ids'], inputs['classification_labels'])

            # choose words at random or argmax or based on probability
            fake_action, _ = word_index_selector(fake_vocabulary_logprobs, 'sample_log', update_step=-1)

            fake_input_ids = (my_attention_mask * fake_action +
                              (1 - my_attention_mask) * inputs['input_ids']).long()

            fake_input_ids = fake_input_ids.view(-1, fake_input_ids.shape[-1])
            # real_input_ids = inputs['input_ids'].view(-1, inputs['input_ids'].shape[-1])

            oracle_vocabulary_logprobs = oracleM(masked_input_ids, inputs['attention_mask'],
                                                 inputs['token_type_ids'], inputs['classification_labels'])
            oracle_vocabulary_logprobs = oracle_vocabulary_logprobs.squeeze()

            oracle_logprobs = oracle_vocabulary_logprobs.gather(dim=2, index=fake_input_ids.unsqueeze(-1)).squeeze()
            my_attention_mask = inputs['my_attention_mask'].view(-1, inputs['my_attention_mask'].shape[-1])

            current_perplexities = (-1. * torch.sum(oracle_logprobs * my_attention_mask, dim=-1)) / torch.sum(my_attention_mask, dim=-1)

            if all_saved_data['all_perplexities'] is None:
                all_saved_data['all_perplexities'] = current_perplexities
            else:
                all_saved_data['all_perplexities'] = torch.cat((all_saved_data['all_perplexities'], current_perplexities), dim=0)

            # assert fake_input_ids.shape[0] == real_input_ids.shape[0]
            # for i in range(fake_input_ids.shape[0]):
            #     TODO update this to refelct maskgan's way of doing this
                # for n in num_of_grams:
                #     all_saved_data['fake {}-gram counter'.format(n)].update(list(ngrams(fake_input_ids[i, :].detach().cpu().tolist(), n)))
                #     all_saved_data['real {}-gram counter'.format(n)].update(list(ngrams(real_input_ids[i, :].detach().cpu().tolist(), n)))

            if all_saved_data['all_fake_input_ids'] is None:
                all_saved_data['all_fake_input_ids'] = fake_input_ids.detach().cpu()
            else:
                all_saved_data['all_fake_input_ids'] = torch.cat((all_saved_data['all_fake_input_ids'],
                                                                  fake_input_ids.detach().cpu()), dim=0)

    all_saved_data['all_real_input_ids'] = eval_dataset.tensors[0].view(-1, eval_dataset.tensors[0].shape[-1])

    return all_saved_data


def write_out_discriminator_eval(all_saved_data, tokenizer, discriminator_filename):
    num_of_grams = [2, 3, 4]
    n_samples = 100
    n_words = 20
    results = {}

    for _ in range(n_samples):
        sentence_ind = torch.LongTensor(1).random_(all_saved_data['all_real_input_ids'].shape[0]).item()

        real_sentence = tokenizer.decode(all_saved_data['all_real_input_ids'][sentence_ind, :].squeeze()).split()
        fake_sentence = tokenizer.decode(all_saved_data['all_fake_input_ids'][sentence_ind, :].squeeze()).split()

        word_ind = torch.LongTensor(1).random_(min(len(real_sentence), len(fake_sentence)) - n_words).item()

        for n in num_of_grams:
            all_saved_data['fake {}-gram counter'.format(n)].update(list(ngrams(fake_sentence[word_ind:(word_ind + n_words)], n)))
            all_saved_data['real {}-gram counter'.format(n)].update(list(ngrams(real_sentence[word_ind:(word_ind + n_words)], n)))

    for n in num_of_grams:
        results['real {}-gram'.format(n)] = len(all_saved_data['real {}-gram counter'.format(n)]) / sum(
            all_saved_data['real {}-gram counter'.format(n)].values())
        results['fake {}-gram'.format(n)] = len(all_saved_data['fake {}-gram counter'.format(n)]) / sum(
            all_saved_data['fake {}-gram counter'.format(n)].values())

    m, s = torch.std_mean(all_saved_data['all_perplexities'])
    results['perplexity mean'] = m.item()
    results['perplexity one standard deviation'] = s.item()

    bleu_score = corpus_bleu(all_saved_data['all_real_input_ids'].tolist(), all_saved_data['all_fake_input_ids'].tolist())

    results['bleu4 score'] = bleu_score


    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    real_sentences = [tokenizer.decode(ids) for ids in all_saved_data['all_real_input_ids'].tolist()]
    fake_sentences = [tokenizer.decode(ids) for ids in all_saved_data['all_fake_input_ids'].tolist()]

    real_embed_graph = embed(real_sentences)
    fake_embed_graph = embed(fake_sentences)

    distance_graph = tfgan.eval.classifier_metrics.frechet_classifier_distance_from_activations(real_embed_graph,
                                                                                                fake_embed_graph)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        distance_fid = session.run(distance_graph)

    results['fid score'] = distance_fid

    if results:
        with open(discriminator_filename, 'w') as df:
            for description, value in results.items():
                df.write('{}: {:.4f}\n'.format(description, value))

                logger.info('{}: {:.4f}\n'.format(description, value))

    return -1


# to evaluate mode collapse sample bigrams, trigrams and quadgrams and calculate the unique number of each, should be high, compare to original sentences

# can look at perplexity in the form of using my generator to grade unseen test sentences or like seqgan which use an oracle with their own generated words (geometric mean)

# frechet distance between embeddings of generated sentences and unseen sentences

# maybe bleu score of generated sentences and unseen sentences

if __name__ == '__main__':
    main()

