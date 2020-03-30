import torch
import torch.nn as nn
import os, glob
import logging

from utils_classifier import classifier_models_and_config_classes
from utils_attention import attention_models_and_config_classes
from utils_generator import generator_models_and_config_classes

# logging
logger = logging.getLogger(__name__)

# generally setting up the models with initial weights
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(-0.01, 0.01)
        m.bias.data.fill_(0.01)

def inititalize_models(args, tokenizer):
    generator_config_class, generator_model_class = generator_models_and_config_classes[args.generator_model_type]
    classifier_config_class, classifier_model_class = classifier_models_and_config_classes[args.classifier_model_type]
    attention_config_class, attention_model_class = attention_models_and_config_classes[args.attention_model_type]

    attention_config_dicts = {'PMI': {'tokenizer': tokenizer,
                                      'window_size': args.attention_window_size,
                                      'max_attention_words': args.max_attention_words,
                                      },
                              'random': {},
                              'essential': {'mu_p': args.essential_mu_p,
                                            'mask_id': tokenizer.mask_token_id},
                              }
    generator_config_dicts = {'seq': {'pretrained_model_name_or_path': 'seq',
                                      'input_dim': tokenizer.vocab_size,
                                      },
                              'bert': {'pretrained_model_name_or_path': args.generator_model_name},
                              'roberta': {'pretrained_model_name_or_path': args.generator_model_name},
                              'xlmroberta': {'pretrained_model_name_or_path': args.generator_model_name},
                              'albert': {'pretrained_model_name_or_path': args.generator_model_name},
                              }
    classifier_config_dicts = {'linear': {'num_choices': 4,
                                          'in_features': args.max_length,
                                          'hidden_features': 100,
                                          'vocab_size': tokenizer.vocab_size,
                                          'embedding_dimension': 10,},
                               'bert': {'pretrained_model_name_or_path': args.classifier_model_name,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'roberta': {'pretrained_model_name_or_path': args.classifier_model_name,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC'},
                               'albert': {'pretrained_model_name_or_path': args.classifier_model_name,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC',
                                        'output_hidden_states': True},
                               }

    logger.info('Establishing config classes.')
    attention_config = attention_config_class.from_pretrained(**attention_config_dicts[args.attention_model_type])
    generator_config = generator_config_class.from_pretrained(**generator_config_dicts[args.generator_model_type])
    classifier_config = classifier_config_class.from_pretrained(**classifier_config_dicts[args.classifier_model_type])

    attention_model_dicts = {'PMI': {'config': attention_config},
                             'random': {},
                             'essential': {'config': attention_config},
                             }
    generator_model_dicts = {'seq': {'config': generator_config},
                              'bert': {'pretrained_model_name_or_path': args.generator_model_name,
                                       'config': generator_config},
                              'roberta': {'pretrained_model_name_or_path': args.generator_model_name,
                                       'config': generator_config},
                              'xlmroberta': {'pretrained_model_name_or_path': args.generator_model_name,
                                       'config': generator_config},
                              'albert': {'pretrained_model_name_or_path': args.generator_model_name,
                                       'config': generator_config},
                              }
    classifier_model_dicts = {'linear':{'config': classifier_config},
                              'bert': {'pretrained_model_name_or_path': args.classifier_model_name,
                                       'config': classifier_config},
                              'roberta': {'pretrained_model_name_or_path': args.classifier_model_name,
                                       'config': classifier_config},
                              'xlmroberta': {'pretrained_model_name_or_path': args.classifier_model_name,
                                       'config': classifier_config},
                              'albert': {'pretrained_model_name_or_path': args.classifier_model_name,
                                       'config': classifier_config},
                              }

    logger.info('Establishing model classes')
    attentionM = attention_model_class.from_pretrained(**attention_model_dicts[args.attention_model_type])
    generatorM = generator_model_class.from_pretrained(**generator_model_dicts[args.generator_model_type])
    classifierM = classifier_model_class.from_pretrained(**classifier_model_dicts[args.classifier_model_type])

    # apply initial weights
    generatorM.apply(init_weights)
    # attentionM.apply(init_weights)
    classifierM.apply(init_weights)

    # move to proper device based on if gpu is available
    logger.info('Loading models to {}'.format(args.device))
    generatorM.to(args.device)
    attentionM.to(args.device)
    classifierM.to(args.device)

    return generatorM, attentionM, classifierM


def save_models(args, checkpoint, generatorM, classifierM):
    output_dir_generator = os.path.join(args.output_dir, '{}-generator-{}'.format(args.transformer_name, checkpoint))
    if not os.path.exists(output_dir_generator):
        os.makedirs(output_dir_generator)
    generator_model_to_save = generatorM.module if hasattr(generatorM, 'module') else generatorM
    if hasattr(generator_model_to_save, 'save_pretrained'):
        generator_model_to_save.save_pretrained(output_dir_generator)
        logger.info('Saving generator model checkpoint to {}'.format(output_dir_generator))
    else:
        logger.info('Not saving generator model.')

    output_dir_classifier = os.path.join(args.output_dir, '{}-classifier-{}'.format(args.transformer_name, checkpoint))
    if not os.path.exists(output_dir_classifier):
        os.makedirs(output_dir_classifier)
    classifier_model_to_save = classifierM.module if hasattr(classifierM, 'module') else classifierM
    if hasattr(classifier_model_to_save, 'save_pretrained'):
        classifier_model_to_save.save_pretrained(output_dir_classifier)
        logger.info('Saving classifier model checkpoint to {}'.format(output_dir_classifier))
    else:
        logger.info('Not saving classifier model.')

    return -1


def load_models(args, tokenizer):


    _, generator_model_class = generator_models_and_config_classes[args.generator_model_type]
    _, classifier_model_class = classifier_models_and_config_classes[args.classifier_model_type]
    attention_config_class, attention_model_class = attention_models_and_config_classes[args.attention_model_type]

    # load model
    model_folders = glob.glob(os.path.join(args.output_dir, '{}_*'.format(args.transformer_name)))
    assert len(model_folders) > 0, 'No model parameters found'
    indices_dash = [len(mf) - mf[::-1].index('-') for mf in model_folders]

    checkpoints = list(set([int(mf[id:]) for mf, id in zip(model_folders, indices_dash)]))

    if not args.evaluate_all_models:
        checkpoints = [max(checkpoints)]

    models_checkpoints = []
    for cp in checkpoints:
        logger.info('Loading model using checkpoint {}'.format(cp))
        classifier_folder = glob.glob(os.path.join(args.output_dir, '{}-classifier-{}'.format(args.transformer_name, cp)))
        generator_folder = glob.glob(os.path.join(args.output_dir, '{}-generator-{}'.format(args.transformer_name, cp)))

        assert len(classifier_folder) == len(generator_folder) == 1

        generatorM = generator_model_class.from_pretrained(generator_folder[0], config={})
        classifierM = classifier_model_class.from_pretrained(classifier_folder[0], config={})

        attention_config_dicts = {'PMI': {'tokenizer': tokenizer,
                                          'window_size': args.attention_window_size,
                                          'max_attention_words': args.max_attention_words,
                                          },
                                  'random': {},
                                  'essential': {'mu_p': args.essential_mu_p,
                                                'mask_id': tokenizer.mask_token_id},
                                  }
        attention_config = attention_config_class.from_pretrained(**attention_config_dicts[args.attention_model_type])
        attention_model_dicts = {'PMI': {'config': attention_config},
                                 'random': {},
                                 'essential': {'config': attention_config},
                                 }
        attentionM = attention_model_class.from_pretrained(**attention_model_dicts[args.attention_model_type])

        models_checkpoints.append(((attentionM, generatorM, classifierM), cp))

    return models_checkpoints



