import getpass
import logging
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from transformers import (BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer)


if getpass.getuser() == 'Mitch':
    from utils_real_data import example_loader
    from utils_attention import attention_models
    from utils_embedding_model import feature_loader, load_features, save_features
    from utils_classifier import classifier_models_and_config_classes, flip_labels
    from utils_generator import generator_models_and_config_classes
else:
    from utils_real_data import example_loader
    from utils_attention import attention_models
    from utils_embedding_model import feature_loader, load_features, save_features
    from utils_classifier import classifier_models_and_config_classes, flip_labels
    from utils_generator import generator_models_and_config_classes


# logging
logger = logging.getLogger(__name__)

# hugging face transformers default models, can use pretrained ones though too
TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
    'distilbert': DistilBertTokenizer,
    'albert': AlbertTokenizer,
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


# generally setting up the models with initial weights
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(-0.01, 0.01)
        m.bias.data.fill_(0.01)


# from hf, returns a list of lists from features of a selected field within the choices_features list of dicts
def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def detach_inputs(fake_inputs, inputs):
    assert 'input_ids' in fake_inputs and 'input_ids' in inputs
    assert 'inputs_embeds' in fake_inputs

    fake_inputs = {k: v.detach() if hasattr(v, 'detach') else v for k, v in fake_inputs.items()}
    inputs = {k: v.detach() if hasattr(v, 'detach') else v for k, v in inputs.items()}

    return fake_inputs, inputs


# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# returns a list of length num_choices with each entry a length four list with a 1 in the correct response spot and 0s elsewhere
def label_map(labels, num_choices):

    def label_list(target, new_target, switch_index, num_choices):
        l = [target]*num_choices
        l[switch_index] = new_target
        return l

    answers = [label_list(0, 1, lab, num_choices) for lab in labels]
    return answers


def load_and_cache_features(args, tokenizer, subset):
    assert subset in ['train', 'dev', 'test']

    cached_features_filename = os.path.join(args.cache_dir, '{}_{}_{}'.format(subset, args.tokenizer_name, args.max_length))
    if os.path.exists(cached_features_filename) and not args.overwrite_cache_dir:
        logger.info('Loading features from ({})'.format(cached_features_filename))
        features = load_features(cached_features_filename)
    else:
        # load in examples and features for training
        logger.info('Creating examples and features from ({})'.format(args.data_dir))
        examples = example_loader(args, subset=subset)
        features = feature_loader(args, tokenizer, examples)

        assert save_features(features, cached_features_filename) == -1

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_token_type_mask = torch.tensor(select_field(features, 'token_type_mask'), dtype=torch.long)
    all_attention_mask = torch.tensor(select_field(features, 'attention_mask'), dtype=torch.long)
    all_labels = torch.tensor(label_map([f.label for f in features], num_choices=4), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_mask, all_attention_mask, all_labels)

    return dataset


def evaluate(args, classifierM, tokenizer, test=False):
    results = {}

    eval_dataset = load_and_cache_features(args, tokenizer, subset='test' if test else 'dev')

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=min(args.batch_size, 100))

    logger.info('Starting Evaluation!')
    logger.info('Number of examples: {}'.format(len(eval_dataset)))

    eval_loss = 0.
    all_predictions = None
    all_labels = None
    num_steps = 0

    for batch in tqdm(eval_dataloader, 'Evaluating'):
        classifierM.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      #'my_attention_mask': batch[3],
                      'labels': batch[4],
                      }
            eval_predictions, eval_error = classifierM(**inputs)

            eval_loss += eval_error.mean().item()

        if all_predictions is None:
            all_predictions = eval_predictions.detach().cpu().numpy()
            all_labels = inputs['labels'].detach().cpu().numpy()
        else:
            all_predictions = np.append(all_predictions, eval_predictions.detach().cpu().numpy(), axis=0)
            all_labels = np.append(all_labels, inputs['labels'].detach().cpu().numpy(), axis=0)

        num_steps += 1

    eval_loss = eval_loss / num_steps
    all_predictions = np.argmax(all_predictions, axis=1)
    all_labels = np.argmax(all_labels, axis=1)
    accuracy = accuracy_score(all_labels, all_predictions)
    num_correct = accuracy_score(all_labels, all_predictions, normalize=False)

    results['accuracy'] = accuracy
    results['number correct'] = num_correct
    results['number of examples'] = len(eval_dataset)
    results['loss'] = round(eval_loss, 5)

    logger.info('After evaluating:')
    for key, val in results.items():
        logger.info('The {} is {}'.format(key, round(val, 3)))

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='Folder where the data is being stored')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='Folder where output should be sent')
    parser.add_argument('--cache_dir', default=None, type=str, required=True,
                        help='Folder where saved models will be written')
    parser.add_argument('--transformer_name', default=None, type=str, required=True,
                        help='Name of the transformer used in tokenizing in {}'.format(', '.join(list(TOKENIZER_CLASSES.keys()))))
    parser.add_argument('--tokenizer_name', default=None, type=str, required=True,
                        help='Name of the tokenizer to use from transformers package from a pretrained hf model')

    # Optional
    parser.add_argument('--generator_model_type', default='seq', type=str,
                        help='Type of the generator model to use from {}'.format(', '.join(list(generator_models_and_config_classes.keys()))))
    parser.add_argument('--generator_model_name_or_path', default=None, type=str,
                        help='Name or path to generator model.')
    parser.add_argument('--classifier_model_type', default='linear', type=str,
                        help='Name of the classifier model to use')
    parser.add_argument('--classifier_model_name_or_path', default=None, type=str,
                        help='Name or path to classifier model.')
    parser.add_argument('--attention_model_type', default='linear', type=str,
                        help='Name of attention model to use')
    parser.add_argument('--attention_model_name_or_path', default=None, type=str,
                        help='Name or path to attention model.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Size of each batch to be used in training')
    parser.add_argument('--max_length', type=int, default=512,
                        help='The maximum length of the sequences allowed. This will induce cutting off or padding')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='After each epoch test model on evaluation set')
    parser.add_argument('--evaluate_during_training_steps', type=int, default=25,
                        help='Evaluate during training each time after this many steps')
    parser.add_argument('--cutoff', type=int, default=None,
                        help='Stop example collection at this number')
    parser.add_argument('--do_randomize', action='store_true',
                        help='Randomize input')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Tokenizer converts everything to lower case')

    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of epochs to run training')
    parser.add_argument('--learning_rate_classifier', default=1e-4, type=float,
                        help='Learning rate of the classifier to be used in Adam optimization')
    parser.add_argument('--learning_rate_generator', default=1e-4, type=float,
                        help='Learning rate of the generator to be used in Adam optimization')
    parser.add_argument('--epsilon_classifier', default=1e-8, type=float,
                        help='Epsilon of classifier for Adam optimizer')
    parser.add_argument('--epsilon_generator', default=1e-8, type=float,
                        help='Epsilon of generator for Adam optimizer')
    # TODO do some annealing with this min_temperature on gumbel softmax
    parser.add_argument('--min_temperature', default=.5, type=float,
                        help='Minimum temperature for annealing')
    parser.add_argument('--save_steps', default=50, type=int,
                        help='After this many steps save models')
    parser.add_argument('--do_evaluate', action='store_true',
                        help='Use models on evaluation dataset')
    parser.add_argument('--do_not_train', action='store_true',
                        help='Train models on training dataset')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use a gpu')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the output directory')
    parser.add_argument('--overwrite_cache_dir', action='store_true',
                        help='Overwrite the cached models directory')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Random seed for reproducibility')

    if not getpass.getuser() == 'Mitch':
        args = parser.parse_args()
    else:
        class Args(object):
            def __init__(self):
                self.data_dir = '../ARC/ARC-with-context/'
                self.output_dir = 'output/'
                self.cache_dir = 'saved/'
                self.tokenizer_name = 'albert-base-v2'
                self.generator_model_type = 'seq'
                self.generator_model_name_or_path = 'albert-base-v2'
                self.classifier_model_type = 'albert'
                self.classifier_model_name_or_path = 'albert-base-v2'
                self.attention_model_type = 'linear'
                self.attnetion_model_name_or_path = None
                self.transformer_name = 'albert'
                self.evaluate_during_training = False
                self.cutoff = 50
                self.do_randomize = False
                self.epochs = 3
                self.learning_rate_classifier = 1e-4
                self.learning_rate_generator = 1e-4
                self.epsilon_classifier = 1e-9
                self.epsilon_generator = 1e-9
                self.do_evaluate = False
                self.do_not_train = False
                self.use_gpu = False
                self.overwrite_output_dir = True
                self.overwrite_cache_dir = False
                self.seed = 1234
                self.max_length = 512
                self.batch_size = 5
                self.do_lower_case = True
                self.save_steps = 20

        args = Args()

    # Setup logging
    num_logging_files = len(glob.glob('logging/logging_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/logging_{}'.format(num_logging_files))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise Exception("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))

    for arg, value in sorted(vars(args).items()):
        logging.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    # get just tokenizer for now, maybe in the future can get default models too
    tokenizer_class = TOKENIZER_CLASSES[args.transformer_name]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    dataset = load_and_cache_features(args, tokenizer, 'train')

    # use pytorch data loaders to cycle through the data,
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    # get whether running on cpu or gpu
    device = get_device() if args.use_gpu else torch.device('cpu')
    args.device = device
    logger.info('Using device {}'.format(args.device))

    generator_config_class, generator_model_class = generator_models_and_config_classes[args.generator_model_type]
    classifier_config_class, classifier_model_class = classifier_models_and_config_classes[args.classifier_model_type]

    generator_config_dicts = {'seq': {'pretrained_model_name_or_path': 'seq',
                                      'input_dim': tokenizer.vocab_size,
                                      },
                              'bert': {'pretrained_model_name_or_path': args.generator_model_name_or_path},
                              'roberta': {'pretrained_model_name_or_path': args.generator_model_name_or_path},
                              'xlmroberta': {'pretrained_model_name_or_path': args.generator_model_name_or_path},
                              'albert': {'pretrained_model_name_or_path': args.generator_model_name_or_path},
                              }
    classifier_config_dicts = {'linear': {'num_choices': 4,
                                          'in_features': args.max_length,
                                          'hidden_features': 100,
                                          'vocab_size': tokenizer.vocab_size,
                                          'embedding_dimension': 10,},
                               'bert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'roberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'albert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               }

    logger.info('Establishing config classes.')
    generator_config = generator_config_class.from_pretrained(**generator_config_dicts[args.generator_model_type])
    classifier_config = classifier_config_class.from_pretrained(**classifier_config_dicts[args.classifier_model_type])

    generator_model_dicts = {'seq': {'config': generator_config},
                              'bert': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              'roberta': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              'xlmroberta': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              'albert': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              }
    classifier_model_dicts = {'linear':{'config': classifier_config},
                              'bert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              'roberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              'albert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              }

    logger.info('Establishing model classes')
    generatorM = generator_model_class.from_pretrained(**generator_model_dicts[args.generator_model_type])
    classifierM = classifier_model_class.from_pretrained(**classifier_model_dicts[args.classifier_model_type])

    # create network instances and initialize weights
    # generatorM = generator_models[generator_name](config)  # input features
    attentionM = attention_models[args.attention_model_type]()  # input features
    # classifierM = classifier_models[classifier_name](config)  # input features, num choices

    # apply initial weights
    generatorM.apply(init_weights)
    # attentionM.apply(init_weights)
    classifierM.apply(init_weights)

    # move to proper device based on if gpu is available
    logger.info('Loading models to {}'.format(args.device))
    generatorM.to(args.device)
    attentionM.to(args.device)
    classifierM.to(args.device)

    if args.use_gpu:
        logger.info('All models uploaded to {}, total memory is {} GB cached, and {} GB allocated.'.format(args.device,
                                                                                                           torch.cuda.memory_allocated(args.device),
                                                                                                           torch.cuda.memory_cached(args.device)))
        logger.info('The number of gpus available is {}'.format(torch.cuda.device_count()))

    # optimizers
    classifierO = AdamW(classifierM.parameters(), lr=args.learning_rate_classifier, eps=args.epsilon_classifier)
    generatorO = AdamW(generatorM.parameters(), lr=args.learning_rate_generator, eps=args.epsilon_generator)
    # attentionO = AdamW(attentionM.parameters(), lr=learning_rateG, eps=epsilonG)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    best_dev_acc = 0.0

    logger.info('Starting to train!')
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for iterate, batch in enumerate(epoch_iterator):
            logger.info('Epoch: {} Iterate: {}'.format(epoch, iterate))

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'my_attention_mask': batch[3],
                      'labels': batch[4],
                      }

            # Train generator
            generatorM.train()
            # attentionM.train()

            # this changes the 'my_attention_masks' input to highlight which words should be changed
            fake_inputs = attentionM(**inputs)
            logger.info('Attention success!')

            # this changes the 'input_ids' based on the 'my_attention_mask' input to generate words to fool classifier
            fake_inputs = {k: v.to(args.device) for k, v in fake_inputs.items()}
            fake_inputs = generatorM(**fake_inputs)
            logger.info('Generator success!')

            # flip labels to represent the wrong answers are actually right
            fake_inputs = flip_labels(**fake_inputs)

            # get the predictions of which answers are the correct pairing from the classifier
            fake_inputs = {k: v.to(args.device) for k, v in fake_inputs.items()}
            predictions, errorG = classifierM(**fake_inputs)
            logger.info('Generator classification success')

            if errorG is None:
                logger.warning('ErrorG is None!')
                raise Exception('ErrorG is None!')

            # based on the loss function update the parameters within the generator/ attention model
            errorG.backward()

            # print('attention model')
            # print(list(attentionM.parameters())[0].grad)
            # print(torch.max(list(attentionM.parameters())[0].grad))
            print('generator model')
            for i in range(len(list(generatorM.parameters()))):
                print(i)
                print(list(generatorM.parameters())[i].grad)
                # print(torch.max(list(generatorM.parameters())[i].grad))
            print('classifier model')
            for i in range(len(list(classifierM.parameters()))):
                print(i)
                print(list(classifierM.parameters())[i].grad)
            # print(torch.max(list(classifierM.parameters())[0].grad))
            # print('*****************************************************************')
            if all([list(generatorM.parameters())[i].grad is None for i in range(len(list(generatorM.parameters())))]):
                raise Exception('There is no gradient parameters for the generator (all None) in epoch {} iteration {}!'.format(epoch, iterate))
            if any([torch.max(torch.abs(list(generatorM.parameters())[i].grad)) == 0 for i in range(len(list(generatorM.parameters()))) if list(generatorM.parameters())[i].grad is not None]):
                raise Exception('There is some zero gradient parameters for the generator in epoch {} iteration {}!'.format(epoch, iterate))

            # Update generatorM parameters
            generatorO.step()
            # attentionO.step()
            logger.info('Generator step success!')

            # zero out gradient of networks
            generatorM.zero_grad()
            # attentionM.zero_grad()
            classifierM.zero_grad()

            # Train classifier
            classifierM.train()

            # detach the inputs so the gradient graphs don't reach back, only need them for classifier
            fake_inputs, inputs = detach_inputs(fake_inputs, inputs)

            # see if the classifier can determine difference between fake and real data
            predictions_fake, error_fake = classifierM(**fake_inputs)
            predictions_real, error_real = classifierM(**inputs)
            logger.info('Classifier fake and real data success!')

            if error_fake is None:
                logger.warning('Error_fake is None!')
                raise Exception('Error_fake is None!')
            if error_real is None:
                logger.warning('Error_real is None!')
                raise Exception('Error_real is None!')

            # calculate gradients from each loss functions
            error_fake.backward()
            error_real.backward()

            # print('attention model')
            # print(list(attentionM.parameters())[0].grad)
            # print(torch.max(list(attentionM.parameters())[0].grad))
            # print('generator model')
            # for i in range(len(list(generatorM.parameters()))):
            #     print(i)
            #     print(list(generatorM.parameters())[i].grad)
                # print(torch.max(list(generatorM.parameters())[i].grad))
            # print('classifier model')
            # for i in range(len(list(classifierM.parameters()))):
            #     print(list(classifierM.parameters())[i].grad)
            #     print(torch.max(list(classifierM.parameters())[i].grad))
            # print('*****************************************************************')
            if all([list(classifierM.parameters())[i].grad is None for i in range(len(list(classifierM.parameters())))]):
                raise Exception('There are no gradient parameters for the classifier (all None) in epoch {} iteration {}!'.format(epoch, iterate))
            if any([torch.max(torch.abs(list(classifierM.parameters())[i].grad)) == 0 for i in range(len(list(classifierM.parameters()))) if list(classifierM.parameters())[i].grad is not None]):
                raise Exception('There is some zero gradient parameters for the classifier in epoch {} iteration {}!'.format(epoch, iterate))

            # update classifier parameters
            classifierO.step()
            logger.info('Classifier step success!')

            # zero out gradient of networks
            generatorM.zero_grad()
            # attentionM.zero_grad()
            classifierM.zero_grad()


            # add errors together for logging purposes
            errorD = error_real + error_fake

            # log error for this step
            logger.info('The generator error is {}'.format(round(errorG.detach().item(), 3)))
            logger.info('The classfier error is {}'.format(round(errorD.detach().item(), 3)))

            # save models in cache dir
            if (epoch*args.batch_size + iterate + 1) % args.save_steps == 0 and args.save_steps > 0:
                # TODO test this
                output_dir_generator = os.path.join(args.output_dir, 'checkpoint-generator-{}'.format(epoch*args.batch_size + iterate + 1))
                if not os.path.exists(output_dir_generator):
                    os.makedirs(output_dir_generator)
                generator_model_to_save = generatorM
                if hasattr(generator_model_to_save, 'save_pretrained'):
                    generator_model_to_save.save_pretrained(output_dir_generator)
                    logger.info('Saving generator model checkpoint to {}'.format(output_dir_generator))
                else:
                    logger.info('Not saving generator model.')

                output_dir_classifier = os.path.join(args.output_dir, 'checkpoint-classifier-{}'.format(epoch*args.batch_size + iterate + 1))
                if not os.path.exists(output_dir_classifier):
                    os.makedirs(output_dir_classifier)
                classifier_model_to_save = classifierM
                if hasattr(classifier_model_to_save, 'save_pretrained'):
                    logger.info('Saving classifier model checkpoint to {}'.format(output_dir_classifier))
                    classifier_model_to_save.save_pretrained(output_dir_classifier)
                else:
                    logger.info('Not saving classifier model.')

        if args.evaluate_during_training:
            eval_results = evaluate(args, classifierM, tokenizer, test=False)

            if eval_results['accuracy'] > best_dev_acc:
                best_dev_acc = eval_results['accuracy']
                best_dev_loss = eval_results['loss']
                best_epoch = epoch

        epoch_iterator.close()

    train_iterator.close()

    if args.evaluate_during_training:
        logger.info('The best dev accuracy during training is {}, loss is {} on epoch {}'.format(round(best_dev_acc, 3),
                                                                                                 round(best_dev_loss, 3),
                                                                                                 best_epoch))

    if args.do_evaluate:
        eval_results = evaluate(args, classifierM, tokenizer)
        logger.info('The dev accuracy after training is {}, loss is {}'.format(round(eval_results['accuracy'], 3),
                                                                               round(eval_results['loss'], 3)))




if __name__ == '__main__':
    main()

# TODO one isssue with this is I need to code up how to handle non pytorch models since they won't work with optimizer/loss functionality
# combinations that work: seq to linear, albert to linear,
# combinations that do not work: seq to albert, albert to albert
