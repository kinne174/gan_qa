import getpass
import logging
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
from transformers import AdamW
from tqdm import tqdm, trange
import os
import argparse
import numpy as np

from transformers import (BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, DistilBertConfig, DistilBertTokenizer)


if getpass.getuser() == 'Mitch':
    from utils_real_data import example_loader
    from utils_attention import attention_loader, attention_models
    from utils_embedding_model import feature_loader, load_features, save_features
    from utils_classifier import classifier_models_and_config_classes, flip_labels
    from utils_generator import generator_models_and_config_classes, gumbel_softmax
else:
    from utils_real_data import example_loader
    from utils_attention import attention_loader, attention_models
    from utils_embedding_model import feature_loader, load_features, save_features
    from utils_classifier import classifier_models_and_config_classes, flip_labels
    from utils_generator import generator_models_and_config_classes, gumbel_softmax


# logging
logger = logging.getLogger(__name__)

# hugging face transformers default models, can use pretrained ones though too
TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
    'distilbert': DistilBertTokenizer,
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
    # answers = torch.tensor(answers, dtype=torch.float)
    return answers

def load_and_cache_features(args, tokenizer, subset):
    assert subset in ['train', 'dev', 'test']

    cached_features_filename = os.path.join(args.cache_dir, '{}_{}_{}'.format('train', args.tokenizer_name, args.max_length))
    if os.path.exists(cached_features_filename) and not args.overwrite_cache_dir:
        features = load_features(cached_features_filename)
    else:
        # load in examples and features for training
        examples = example_loader(args, subset=subset, randomize=args.do_randomize, cutoff=args.cutoff)
        features = feature_loader(args, tokenizer, examples, randomize=args.do_randomize)

        assert save_features(features, cached_features_filename) == -1

    # Convert to Tensors and build dataset
    # TODO problem seems to be that autograd cannot handle torch.long but needed for nn.Embeddings
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_token_type_mask = torch.tensor(select_field(features, 'token_type_mask'), dtype=torch.long)
    all_attention_mask = torch.tensor(select_field(features, 'attention_mask'), dtype=torch.long)
    all_labels = torch.tensor(label_map([f.label for f in features], num_choices=4), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_token_type_mask, all_attention_mask, all_labels)

    return dataset


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
                        help='Name of the transformer used in tokenizing')
    parser.add_argument('--tokenizer_name', default=None, type=str, required=True,
                        help='Name of the tokenizer to use from transformers package in {}'.format(', '.join(list(TOKENIZER_CLASSES.keys()))))

    # Optional
    parser.add_argument('--generator_model_type', default='linear', type=str,
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

    use_argparse = False
    if use_argparse:
        args = parser.parse_args()
    else:
        class Args(object):
            def __init__(self):
                self.data_dir = '../ARC/ARC-with-context/'
                self.output_dir = 'output/'
                self.cache_dir = 'saved/'
                self.tokenizer_name = 'bert-base-uncased'
                self.generator_model_type = 'seq'
                self.generator_model_name_or_path = None
                self.classifier_model_type = 'linear'
                self.classifier_model_name_or_path = None
                self.attention_model_type = 'linear'
                self.attnetion_model_name_or_path = None
                self.transformer_name = 'bert'
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
                self.overwrite_output_dir = False
                self.overwrite_cache_dir = False
                self.seed = 1234
                self.cache_features = False
                self.max_length = 512
                self.batch_size = 5
                self.do_lower_case = True


        args = Args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # load in parameters using something like arg parse, for now just set them
    # epochs = 3
    # randomize = False
    # cutoff = 50
    # generator_name = 'linear'
    # attention_name = 'linear'
    # classifier_name = 'linear'
    # learning_rateC = 1e-4
    # epsilonC = 1e-7
    # learning_rateG = 1e-2
    # epsilonG = 1e-5
    # data_dir = '../ARC/ARC-with-context/'
    # output_dir = 'output/'
    # tokenizer_name = 'bert-base-uncased'
    # transformer_model_name = 'bert'
    # do_lower_case = True

    # Set seed
    set_seed(args)

    # get just tokenizer for now, maybe in the future can get default models too
    tokenizer_class = TOKENIZER_CLASSES[args.transformer_name]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    dataset = load_and_cache_features(args, tokenizer, 'train')

    # use pytorch data loaders to cycle through the data,
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    # TODOfixed this should be a dataloader from pytorch instead, has been replaced
    # batch_size = len(examples)//iterations

    # get whether running on cpu or gpu
    device = get_device() if args.use_gpu else torch.device('cpu')

    # Setup logging
    num_logging_files = len(glob.glob('logging/logging_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/logging_{}'.format(num_logging_files))
    logger.info('Using device '.format(device))


    # plan right now is to do similar to https://github.com/diegoalejogm/gans/blob/master/2.%20DC-GAN%20PyTorch-MNIST.ipynb
    # and define the models and optimizers in main before training based on models found in the utils_* files
    # then just define my models in the utils_* and do all the data preparation functions there too
    # TODO one isssue with this is I need to code up how to handle non pytorch models since they won't work with optimizer/loss functionality

    generator_config_class, generator_model_class = generator_models_and_config_classes[args.generator_model_type]
    classifier_config_class, classifier_model_class = classifier_models_and_config_classes[args.classifier_model_type]

    generator_config_dicts = {'linear': {'pretrained_model_name_or_path': 'linear'},
                              'seq': {'pretrained_model_name_or_path': 'seq',
                                      'device': device,
                                      'input_dim': tokenizer.vocab_size,
                                      },
                              'bert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path},
                              'roberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path},
                              'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path},
                              }
    classifier_config_dicts = {'linear': {'num_choices': 4,
                                          'in_features': args.max_length,
                                          'hidden_features': 100,
                                          },
                               'bert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'roberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               }

    generator_config = generator_config_class.from_pretrained(**generator_config_dicts[args.generator_model_type])
    classifier_config = classifier_config_class.from_pretrained(**classifier_config_dicts[args.classifier_model_type])


    generator_model_dicts = {'linear': {'config': generator_config},
                              'seq': {'config': generator_config},
                              'bert': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              'roberta': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              'xlmroberta': {'pretrained_model_name_or_path': args.generator_model_name_or_path,
                                       'config': generator_config},
                              }
    classifier_model_dicts = {'linear':{'config': classifier_config},
                              'bert': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              'roberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name_or_path,
                                       'config': classifier_config},
                              }

    generatorM = generator_model_class.from_pretrained(**generator_model_dicts[args.generator_model_type])
    classifierM = classifier_model_class.from_pretrained(**classifier_model_dicts[args.classifier_model_type])

    # create network instances and initialize weights
    # generatorM = generator_models[generator_name](config)  # input features
    attentionM = attention_models[args.attention_model_type]()  # input features
    # classifierM = classifier_models[classifier_name](config)  # input features, num choices

    # apply initial weights
    # generatorM.apply(init_weights)
    # attentionM.apply(init_weights)
    classifierM.apply(init_weights)

    # print(generatorM)

    # move to proper device based on if gpu is available
    generatorM.to(device)
    attentionM.to(device)
    classifierM.to(device)

    if device == 'cuda':
        logger.info('All models uploaded to {}, total memory is {} GB cached, and {} GB allocated.'.format(device,
                                                                                                           torch.cuda.memory_allocated(device),
                                                                                                           torch.cuda.memory_cached(device)))

    # optimizers
    classifierO = AdamW(classifierM.parameters(), lr=args.learning_rate_classifier, eps=args.epsilon_classifier)
    generatorO = AdamW(generatorM.parameters(), lr=args.learning_rate_generator, eps=args.epsilon_generator)
    # attentionO = AdamW(attentionM.parameters(), lr=learning_rateG, eps=epsilonG)

    train_iterator = trange(int(args.epochs), desc="Epoch")

    for e, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for i, batch in enumerate(epoch_iterator):
            # TODOfixed is there a better way to handle which examples are to be taken as fake and which are to be taken as real?
            # current_examples = examples[batch_size*i:batch_size*(i+1)]
            #
            # real_examples = current_examples[:batch_size//2]
            # fake_examples = current_examples[batch_size//2:]

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'my_attention_mask': batch[3],
                      'labels': batch[4],
                      'device': device,
                      }

            # Train generator
            generatorM.train()
            # attentionM.train()

            # from examples convert to input ids to be run in the neural networks, this changes from words to integers
            # next is to put the inputids into the attention network
            # fake_inputids, fake_features = feature_loader(fake_examples, randomize=randomize)

            # this changes the 'my_attention_masks' input
            fake_inputs = attentionM(**inputs)

            # TODOfixed this should be like an attention thing in a network, not fully sure how to implement it right now though
            # fake_inputids = fake_inputids*attention_masks

            # TODOfixed update context attention masks in fake_features within inputs
            # fake_features = attention_loader(fake_features, attention_masks)

            # this changes the 'input_ids' based on the 'my_attention_mask' input
            # TODOfixed generator should receive as input the context attention masks too
            fake_inputs = generatorM(**fake_inputs)
            # TODOfixed changed_inputids is all zero, probably not right, for some reason batchnorm did it, hopefully
            # this isn't a problem when moving to huggingface transformers

            criterion = nn.MSELoss(reduction='mean')
            loss = criterion(fake_inputs['input_ids'].float(), inputs['input_ids'].float())
            loss.backward()
            print('generator model')
            print(list(generatorM.parameters())[0].grad)
            print(torch.max(list(generatorM.parameters())[0].grad))
            generatorO.step()

            # create labeling based on changing the correct answer to wrong in eyes of the classifier, classifier should
            # be able to determine which questing ending pair is incorrect
            # fake_labels = label_map([ff.label for ff in fake_features], fake=True, num_choices=4)

            # TODOfixed incorporate changed_inputids into the inputs variable

            # TODOfixed labels need to be flipped here
            fake_inputs = flip_labels(**fake_inputs)
            # get the predictions of which answers are the correct pairing from the classifier
            predictions, errorG = classifierM(**fake_inputs)

            # based on the loss function update the parameters within the generator/ attention model
            # TODOfixed should also be the attention model but right now just the generator
            # errorG = loss(predictions, fake_labels)
            errorG.backward()

            # print('attention model')
            # print(list(attentionM.parameters())[0].grad)
            # print(torch.max(list(attentionM.parameters())[0].grad))
            print('generator model')
            print(list(generatorM.parameters())[0].grad)
            print(torch.max(list(generatorM.parameters())[0].grad))
            print('classifier model')
            print(list(classifierM.parameters())[0].grad)
            print(torch.max(list(classifierM.parameters())[0].grad))
            print('*****************************************************************')

            # Update generatorM parameters
            generatorO.step()
            # attentionO.step()

            # zero out gradient of networks
            generatorM.zero_grad()
            attentionM.zero_grad()
            classifierM.zero_grad()

            # Train classifier
            classifierM.train()

            # load in the features and inputids from the real examples
            # real_inputids, real_features = feature_loader(real_examples, randomize=randomize)

            # create the labels
            # real_labels = label_map([rf.label for rf in real_features], fake=False, num_choices=4)

            # classify the inputs, use the changed inputids from before in hopes of improvement
            # real_predictions = classifierM(real_inputids)
            # fake_predictions = classifierM(changed_inputids)

            # find the error from the real predictions and the fake predictions
            # real_error = loss(real_predictions, real_labels)
            # fake_error = loss(fake_predictions, fake_labels)

            predictions_fake, error_fake = classifierM(**fake_inputs)
            predictions_real, error_real = classifierM(**inputs)

            # calculate gradients from each loss functions
            error_fake.backward()
            error_real.backward()

            # update classifier parameters
            classifierO.step()

            # zero out gradient of networks
            generatorM.zero_grad()
            attentionM.zero_grad()
            classifierM.zero_grad()


            # add errors together for logging purposes
            errorD = error_real + error_fake

            # log error for this step
            # TODO write this step to logging file

            if args.evaluate_during_training and (e*args.batch_size + i + 1) % args.evaluate_during_trainig_steps == 0:
                # TODO write evaluation code and put it here
                pass

            # save models in cache dir
            if (e*args.batch_size + i + 1) % args.save_steps == 0 and args.save_steps > 0:
                # TODO save models in cache directory
                pass

        epoch_iterator.close()

    train_iterator.close()

    if args.do_evaluate:
        # TODO write evaluation code
        pass


if __name__ == '__main__':
    main()

# TODOfixed create a psuedo attention network that takes in as input the attention mask and outputs just a few ones from those indices already ones
# TODOfixed fix classifier part of run_gan code
# TODOfixed refer to transformers to create a way to load and save models/ config classes
# TODOfixed create argparse with all necessary options
# TODO map out how autograd can go from generator to classifier
