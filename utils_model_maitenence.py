import torch
import torch.nn as nn
import os, glob
import logging

from utils_classifier import classifier_models_and_config_classes
from utils_attention import attention_models_and_config_classes
from utils_generator import generator_models_and_config_classes
from utils_discriminator import discriminator_models_and_config_classes

# logging
logger = logging.getLogger(__name__)


def inititalize_models(args, tokenizer):
    generator_config_class, generator_model_class = generator_models_and_config_classes[args.generator_model_type]
    classifier_config_class, classifier_model_class = classifier_models_and_config_classes[args.classifier_model_type]
    attention_config_class, attention_model_class = attention_models_and_config_classes[args.attention_model_type]
    discriminator_config_class, discriminator_model_class = discriminator_models_and_config_classes[args.discriminator_model_type]

    # essential, essential-reinforce
    attention_config_dict = {'mu_p': args.essential_mu_p,
                             'mask_id': tokenizer.mask_token_id}

    # roberta, albert, seq, roberta-reinforce
    generator_config_dict = {'pretrained_model_name_or_path': args.generator_model_name}
    if args.generator_model_type in ['seq']:
        generator_config_dict.update({'input_dim': tokenizer.vocab_size})

    # roberta, albert, linear-reinforce, roberta-reinforce, bert-reinforce
    classifier_config_dict = {}
    if args.classifier_model_type in ['roberta', 'albert', 'roberta-reinforce', 'bert-reinforce']:
        classifier_config_dict.update({'pretrained_model_name_or_path': args.classifier_model_name,
                                        'num_labels': 4,
                                        'finetuning_task': 'ARC',
                                        'output_hidden_states': True})
    if args.classifier_model_type in ['linear-reinforce']:
        classifier_config_dict.update({'vocab_size': tokenizer.vocab_size,
                                       'embedding_dimension': args.classifier_embedding_dim,
                                       'hidden_dim': args.classifier_hidden_dim,
                                       'num_layers': args.classifier_num_layers})

    # lstm, lstm-reinforce
    discriminator_config_dict = {'embedding_type': args.discriminator_embedding_type,
                                 'embedding_dim': args.discriminator_embedding_dim,
                                 'hidden_dim': args.discriminator_hidden_dim,
                                 'num_layers': args.discriminator_num_layers,
                                 'vocab_size': tokenizer.vocab_size,}

    logger.info('Establishing config classes.')
    attention_config = attention_config_class.from_pretrained(**attention_config_dict)
    generator_config = generator_config_class.from_pretrained(**generator_config_dict)
    classifier_config = classifier_config_class.from_pretrained(**classifier_config_dict)
    discriminator_config = discriminator_config_class(**discriminator_config_dict)

    # essential, essential-reinforce
    attention_model_dict = {'config': attention_config}
    if args.attention_model_type in ['essential']:
        attention_model_dict.update({'device': args.device})

    # roberta, albert, seq, roberta-reinforce, bert-reinforce
    generator_model_dict = {'config': generator_config}
    if args.generator_model_type in ['roberta', 'albert']:
        generator_model_dict.update({'device': args.device})
    if args.generator_model_type in ['roberta', 'albert', 'roberta-reinforce', 'bert-reinforce']:
        generator_model_dict.update({'pretrained_model_name_or_path': args.generator_model_name})

    # roberta, albert, linear-reinforce, roberta-reinforce, 'bert-reinforce'
    classifier_model_dict = {'config': classifier_config,
                             'pretrained_model_name_or_path': args.classifier_model_name}
    if args.classifier_model_type in ['roberta', 'albert']:
        classifier_model_dict.update({'device': args.device})

    # lstm, lstm-reinforce
    discriminator_model_dict = {'pretrained_model_path': args.discriminator_model_name,
                                'config': discriminator_config}

    logger.info('Establishing model classes')
    attentionM = attention_model_class.from_pretrained(**attention_model_dict)
    generatorM = generator_model_class.from_pretrained(**generator_model_dict)
    classifierM = classifier_model_class.from_pretrained(**classifier_model_dict)
    discriminatorM = discriminator_model_class.from_pretrained(**discriminator_model_dict)

    # move to proper device based on if gpu is available
    logger.info('Loading models to {}'.format(args.device))
    generatorM.to(args.device)
    attentionM.to(args.device)
    classifierM.to(args.device)
    discriminatorM.to(args.device)

    return generatorM, attentionM, classifierM, discriminatorM


def save_models(args, checkpoint, generatorM, classifierM, discriminatorM):
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

    output_dir_discriminator = os.path.join(args.output_dir, '{}-discriminator-{}'.format(args.transformer_name, checkpoint))
    if not os.path.exists(output_dir_discriminator):
        os.makedirs(output_dir_discriminator)
    discriminatorM.save_pretrained(output_dir_discriminator)
    logger.info('Saving discriminator model checkpoint to {}'.format(output_dir_discriminator))

    logger.info('Models saved!')

    return -1


def load_models(args, tokenizer):

    _, generator_model_class = generator_models_and_config_classes[args.generator_model_type]
    _, classifier_model_class = classifier_models_and_config_classes[args.classifier_model_type]
    attention_config_class, attention_model_class = attention_models_and_config_classes[args.attention_model_type]

    # load model
    model_folders = glob.glob(os.path.join(args.output_dir, '{}-*-*'.format(args.transformer_name)))
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

        # roberta, albert, seq, roberta-reinforce
        generator_model_dict = {'config': generator_folder[0]}
        if args.generator_model_type in ['roberta', 'albert']:
            generator_model_dict.update({'device': args.device})
        if args.generator_model_type in ['roberta', 'albert', 'roberta-reinforce', 'bert-reinforce']:
            generator_model_dict.update({'pretrained_model_name_or_path': generator_folder[0]})

        # roberta, albert, linear-reinforce, roberta-reinforce
        classifier_model_dict = {'config': classifier_folder[0],
                                 'pretrained_model_name_or_path': classifier_folder[0]}
        if args.classifier_model_type in ['roberta', 'albert']:
            classifier_model_dict.update({'device': args.device})

        generatorM = generator_model_class.from_pretrained(**generator_model_dict)
        classifierM = classifier_model_class.from_pretrained(**classifier_model_dict)

        if not generatorM.model.config.output_hidden_states:
            generatorM.model.config.output_hidden_states = True

        attention_config_dict = {'mu_p': args.essential_mu_p,
                                 'mask_id': tokenizer.mask_token_id}
        attention_config = attention_config_class.from_pretrained(**attention_config_dict)

        # essential, essential-reinforce
        attention_model_dict = {'config': attention_config}
        if args.attention_model_type in ['essential']:
            attention_model_dict.update({'device': args.device})
        attentionM = attention_model_class.from_pretrained(**attention_model_dict)

        models_checkpoints.append(((attentionM, generatorM, classifierM), cp))

    return models_checkpoints



