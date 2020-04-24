import os, glob
import torch
import torch.nn as nn
import logging
from sklearn.metrics import accuracy_score
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import nltk
from nltk import ngrams
from nltk.translate.bleu_score import corpus_bleu
from collections import Counter

from transformers import RobertaTokenizer

from run_reinforce import load_and_cache_features, get_device
from utils_generator import generator_models_and_config_classes
from utils_attention import attention_models_and_config_classes
from utils_reinforce import word_index_selector

logger = logging.getLogger(__name__)

# evaluation for generator mainly, classification can be done with pre training on other models and comparing performance, possibly with other datasets or GLUE benchmarks
def main():

    class Args(object):
        def __init__(self):
            self.generator_directory = ''
            self.cache_dir = 'saved/'
            self.transformer_name = 'roberta'
            self.tokenizer_name = 'roberta-base'
            self.output_dir = 'output/'

            self.mu_p = 0.30

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
    args.device = 'cuda:5'

    # get tokenizer
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=True)

    _, generator_model_class = generator_models_and_config_classes['roberta-reinforce']
    # roberta, albert, seq, roberta-reinforce
    generator_model_dict = {'config': args.generator_directory,
                            'pretrained_model_name_or_path': args.generator_directory}
    generatorM = generator_model_class.from_pretrained(**generator_model_dict)

    attention_config_class, attention_model_class = attention_models_and_config_classes['roberta-reinforce']
    attention_config_dict = {'mu_p': args.essential_mu_p,
                             'mask_id': tokenizer.mask_token_id}
    attention_config = attention_config_class.from_pretrained(**attention_config_dict)
    attention_model_dict = {'config': attention_config}
    attentionM = attention_model_class.from_pretrained(**attention_model_dict)

    args.test_filename = os.path.join(args.cache_dir, 'test_roberta-base_256_')


def stream_data(args, generatorM, attentionM, tokenizer):
    generatorM.eval()
    attentionM.eval()

    eval_dataset = load_and_cache_features(args, tokenizer, 'test')

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    results = {}

    if args.do_mode_collapse:
        num_of_grams = [2, 3, 4]

        fake_gram_counters = {n: Counter() for n in num_of_grams}
        real_gram_counters = {n: Counter() for n in num_of_grams}

    if args.do_perplexity:
        assert args.perplexity_model is not None

        _, generator_model_class = generator_models_and_config_classes['roberta-reinforce']
        # roberta, albert, seq, roberta-reinforce
        generator_model_dict = {'config': args.perplexity_generator_directory,
                                'pretrained_model_name_or_path': args.perplexity_generator_directory}
        oracleM = generator_model_class.from_pretrained(**generator_model_dict)

        all_perplexities = None

    if args.do_bleu:
        all_fake_input_ids = None

    with torch.no_grad():
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
            fake_action = word_index_selector(fake_vocabulary_logprobs, 'sample-log')

            fake_input_ids = (my_attention_mask * fake_action +
                              (1 - my_attention_mask) * inputs['input_ids']).long()

            fake_input_ids = fake_input_ids.view(-1, fake_input_ids.shape[-1])
            real_input_ids = inputs['input_ids'].view(-1, inputs['input_ids'].shape[-1])

            if args.do_perplexity:
                oracle_vocabulary_logprobs = oracleM(masked_input_ids, inputs['attention_mask'],
                                                     inputs['token_type_ids'], inputs['classification_labels'])
                oracle_vocabulary_logprobs = oracle_vocabulary_logprobs.view(-1, oracle_vocabulary_logprobs.shape[:-2])

                oracle_probs = oracle_vocabulary_logprobs.gather(dim=2, index=fake_input_ids.unsqueeze(-1)).squeeze()
                my_attention_mask = inputs['my_attention_mask'].view(-1, inputs['my_attention_mask'].shape[-1])

                current_perplexities = torch.prod(oracle_probs * my_attention_mask + (1 - my_attention_mask), dim=-1).squeeze()
                for i in current_perplexities:
                    current_perplexities[i] = current_perplexities[i] ** (-1 * torch.sum(my_attention_mask[i, :],dtype=torch.int))

                if all_perplexities is None:
                    all_perplexities = current_perplexities
                else:
                    all_perplexities = torch.cat((all_perplexities, current_perplexities), dim=0)

            if args.do_mode_collapse:
                assert fake_input_ids.shape[0] == real_input_ids.shape[0]
                for i in range(fake_input_ids.shape[0]):
                    for n in num_of_grams:
                        fake_gram_counters[n].update(list(ngrams(fake_input_ids[i, :], n)))
                        real_gram_counters[n].update(list(ngrams(real_input_ids[i, :], n)))

            if args.do_bleu:
                if all_fake_input_ids is None:
                    all_fake_input_ids = fake_input_ids.detach().cpu()
                else:
                    all_fake_input_ids = torch.cat((all_fake_input_ids, fake_input_ids.detach().cpu()), dim=0)

        if args.do_mode_collapse:
            for n in num_of_grams:
                results['fake {}-gram'.format(n)] = len(fake_gram_counters[n]) / sum(fake_gram_counters[n].values())
                results['real {}-gram'.format(n)] = len(real_gram_counters[n]) / sum(real_gram_counters[n].values())

        if args.do_perplexity:
            m, s = torch.std_mean(all_perplexities)
            results['perplexity mean'] = m.item()
            results['perplexity one standard deviation'] = s.item()

        if args.do_bleu:
            all_real_input_ids = eval_dataset.tensors[0]
            bleu_score = corpus_bleu(all_real_input_ids.tolist(), all_fake_input_ids.tolist())

            results['bleu4 score'] = bleu_score

# to evaluate mode collapse sample bigrams, trigrams and quadgrams and calculate the unique number of each, should be high, compare to original sentences

# can look at perplexity in the form of using my generator to grade unseen test sentences or like seqgan which use an oracle with their own generated words (geometric mean)

# frechet distance between embeddings of generated sentences and unseen sentences

# maybe bleu score of generated sentences and unseen sentences

if __name__ == '__main__':
    main()

