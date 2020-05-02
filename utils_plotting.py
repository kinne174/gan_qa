import json
import os
import glob
import matplotlib.pyplot as plt
import torch
from more_itertools import pairwise
import numpy as np
import statistics
import math


def gather_data_from_logging(args, logging_filename):
    # Things I want to plot:
    #  fid, ngrams, perplexity, rewards, discriminator errors, classifier errors
    # logging_filename = os.path.join(args.logging_dir, logging_filename)
    assert os.path.exists(logging_filename)

    # how do I want to keep track of hyperparameters?
    # probably make a folder for each logging file, store data, graphs and relevant hyperparameters, then later can go through and average data to get better graphs
    hyperparameters = {'accumulation_steps': int,
                       'classification_start': float,
                       'classification_stop': float,
                       'domain_words': str,
                       'epochs': int,
                       'essential_mu_p': float,
                       'lambda_baseline': float,
                       'reinforce_gamma': float,
                       'rewards_start': float,
                       'rewards_stop': float,
                       'seed': int,
                       'transformer_name': str}
    all_hyperparemeters_found = False

    # first store a list (epoch number, batch iterate, value) where batch iterate is reset to zero after epoch changes
    # in post can go through each one and change epoch number, batch iterate to an x value based on epoch + batch/total batches in epoch
    training_values = {'Generator rewards': [],
                       'Classifier real error': [],
                       'Classifier fake error': [],
                       'Discriminator real error': [],
                       'Discriminator fake error': [],
                       'Running Classifier classification': [],
                       'Running Generator classification': [],
                       'Running Generator discimination': [],
                       'Runing Discriminator discrimination': [],  # real discriminator
                        }

    def AssignType(s, t):
        if t is float:
            return float(s)
        elif t is int:
            return int(s)
        elif t is str:
            return s
        else:
            raise NotImplementedError

    current_epoch = -1
    batch_iterate = 0

    # for the file in the logging files
    # keep track of errors and which epoch/ batch they occured
    with open(logging_filename, 'r') as handle:
        for line in handle:
            # find hyperparemeters
            if not all_hyperparemeters_found:
                for hp, t in hyperparameters.items():
                    if hp in line:
                        last_colon_index = len(line) - line[::-1].index(':')  # keep it all characters rather than words
                        hyperparameters[hp] = AssignType(line[last_colon_index+1:].strip(), t)
                        if hp == 'transformer_name':
                            all_hyperparemeters_found = True
                        break

            if 'Running totals' in line:
                epoch_seen = int(line.split()[line.split().index('epoch') + 1][:-1])
                batch_iterate += 1
                if epoch_seen > current_epoch:
                    current_epoch = epoch_seen
                    batch_iterate = 0
                    # continue

                generator_loss_line = next(handle)
                training_values['Generator rewards'].append((epoch_seen, batch_iterate, float(generator_loss_line.split()[-1])))

                classifier_errors_line = next(handle)
                training_values['Classifier fake error'].append((epoch_seen, batch_iterate, float(classifier_errors_line.split()[classifier_errors_line.split().index('fake') + 1])))
                training_values['Classifier real error'].append((epoch_seen, batch_iterate, float(classifier_errors_line.split()[classifier_errors_line.split().index('real') + 1])))

                discriminator_errors_line = next(handle)
                training_values['Discriminator fake error'].append((epoch_seen, batch_iterate, float(discriminator_errors_line.split()[discriminator_errors_line.split().index('fake') + 1])))
                training_values['Discriminator real error'].append((epoch_seen, batch_iterate, float(discriminator_errors_line.split()[discriminator_errors_line.split().index('real') + 1])))

                classifier_line = next(handle)
                training_values['Running Classifier classification'].append((epoch_seen, batch_iterate, float(classifier_line.split()[-1])))

                generator_classification_line = next(handle)
                training_values['Running Generator classification'].append((epoch_seen, batch_iterate, float(generator_classification_line.split()[-1])))

                generator_discrimination_line = next(handle)
                training_values['Running Generator discimination'].append((epoch_seen, batch_iterate, float(generator_discrimination_line.split()[-1])))

                discriminator_line = next(handle)
                training_values['Runing Discriminator discrimination'].append((epoch_seen, batch_iterate, float(discriminator_line.split()[-1])))

    # replace epoch and batch iterate with decimal representative of progress in epoch
    for k, val_list in training_values.items():
        batch_reset = True
        new_val_list = []
        for (e, bi, val) in val_list[::-1]:
            if batch_reset:
                batch_reset = False
                batch_denominator = bi + 1
            new_val_list.append((e + (bi / batch_denominator), val))

            if bi == 0:
                batch_reset = True

        training_values[k] = new_val_list[::-1]

    # save data
    if not os.path.exists(args.current_output_data_dir):
        os.makedirs(args.current_output_data_dir)

    hyperparameter_filename = os.path.join(args.current_output_data_dir, 'hyperparameters.json')
    with open(hyperparameter_filename, 'w') as hyper_filename:
        json.dump(hyperparameters, hyper_filename)

    training_data_filename = os.path.join(args.current_output_data_dir, 'training_data.json')
    with open(training_data_filename, 'w') as training_filename:
        json.dump(training_values, training_filename)


def plot_training_errors(args):

    data_directories = [os.path.join(args.output_data_dir, lf_to_plot) for lf_to_plot in args.logging_filenames_to_plot]

    def add_linear_points(list_of_tuples):
        added_list_of_tuples = []
        for t1, t2 in pairwise(list_of_tuples):
            m = (t2[1] - t1[1]) / (t2[0] - t1[0])
            b = t2[1] - m * t2[0]

            new_xs = np.linspace(start=t1[0], stop=t2[0], endpoint=False, num=20).tolist()
            new_ys = [m*x + b for x in new_xs]

            added_list_of_tuples.extend(list(zip(new_xs, new_ys)))

        added_list_of_tuples.append(list_of_tuples[-1])

        return added_list_of_tuples


    all_generator_rewards = []
    all_real_classifier_error = []
    all_fake_classifier_error = []
    all_real_discriminator_error = []
    all_fake_discriminator_error = []
    for dd in data_directories:
        training_values_filename = os.path.join(dd, 'training_data.json')

        with open(training_values_filename, 'r') as handle:
            training_values = json.load(handle)

        added_generator_rewards = add_linear_points(training_values['Generator rewards'])
        all_generator_rewards.append(added_generator_rewards)

        added_real_classifier_error = add_linear_points(training_values['Classifier real error'])
        all_real_classifier_error.append(added_real_classifier_error)

        added_fake_classifier_error = add_linear_points(training_values['Classifier fake error'])
        all_fake_classifier_error.append(added_fake_classifier_error)

        added_real_discriminator_error = add_linear_points(training_values['Discriminator real error'])
        all_real_discriminator_error.append(added_real_discriminator_error)

        added_fake_discriminator_error = add_linear_points(training_values['Discriminator fake error'])
        all_fake_discriminator_error.append(added_fake_discriminator_error)

    def mean_and_sd_of_values(all_x, list_of_list_of_tuples):
        out_means = []
        out_sd = []

        for x in all_x:
            current_obs = []
            for list_of_tup in list_of_list_of_tuples:
                current_x, current_y = map(list, zip(*list_of_tup))
                closest_ind = np.argmin(np.abs(np.array(current_x) - x))
                if not int(current_x[closest_ind]) == int(x):
                    continue
                current_obs.append(current_y[closest_ind])

            out_means.append(statistics.mean(current_obs))
            if len(current_obs) > 1:
                out_sd.append(statistics.stdev(current_obs))
            else:
                out_sd.append(0)

        out_means = np.array(out_means)
        out_sd = np.array(out_sd)

        return out_means, out_sd

    largest_epoch_seen = int(max([all_generator_rewards[i][-1][0] for i in range(len(all_generator_rewards))]))

    all_x = np.arange(start=0, stop=largest_epoch_seen+1, step=0.1)

    generator_means, generator_sd = mean_and_sd_of_values(all_x, all_generator_rewards)

    real_classifier_means, real_classifier_sd = mean_and_sd_of_values(all_x, all_real_classifier_error)
    fake_classifier_means, fake_classifier_sd = mean_and_sd_of_values(all_x, all_fake_classifier_error)

    real_discriminator_means, real_discriminator_sd = mean_and_sd_of_values(all_x, all_real_discriminator_error)
    fake_discriminator_means, fake_discriminator_sd = mean_and_sd_of_values(all_x, all_fake_discriminator_error)

    ## GENERATOR ##
    fig_generator, ax_generator = plt.subplots()
    ax_generator.plot(all_x, generator_means, color='m')
    ax_generator.fill_between(all_x, (generator_means - generator_sd), (generator_means + generator_sd), color='m', alpha=.2)

    ax_generator.set_xlabel('Epoch')
    ax_generator.set_ylabel('Rewards')
    ax_generator.set_title('Training Rewards for Generator')

    fig_generator.savefig(os.path.join(args.output_plot_dir, 'generator_rewards.png'), bbox_inches='tight')

    ## CLASSIFIER ##
    fig_classifier, ax_classifier = plt.subplots()
    ax_classifier.plot(all_x, real_classifier_means, color='darkgreen', label='Real words')
    ax_classifier.fill_between(all_x, (real_classifier_means - real_classifier_sd), (real_classifier_means + real_classifier_sd),
                               color='darkgreen', alpha=.2)

    ax_classifier.plot(all_x, fake_classifier_means, color='lime', label='Generated words')
    ax_classifier.fill_between(all_x, (fake_classifier_means - fake_classifier_sd), (fake_classifier_means + fake_classifier_sd),
                               color='lime', alpha=.2)

    total_classifier_mean = real_classifier_means + fake_classifier_means
    total_classifier_sd = np.sqrt(real_classifier_sd ** 2 + fake_classifier_sd ** 2)
    ax_classifier.plot(all_x, total_classifier_mean, color='limegreen', label='Total Error')
    ax_classifier.fill_between(all_x, (total_classifier_mean - total_classifier_sd), (total_classifier_mean + total_classifier_sd),
                               color='limegreen', alpha=.2)

    ax_classifier.set_xlabel('Epoch')
    ax_classifier.set_ylabel('Error')
    ax_classifier.set_title('Classifier Error - Real and Generated')

    ax_classifier.legend()
    fig_classifier.savefig(os.path.join(args.output_plot_dir, 'classifier_error.png'), bbox_inches='tight')

    ## DISCRIMINATOR ##
    fig_discriminator, ax_discriminator = plt.subplots()
    ax_discriminator.plot(all_x, real_discriminator_means, color='firebrick', label='Real words')
    ax_discriminator.fill_between(all_x, (real_discriminator_means - real_discriminator_sd), (real_discriminator_means + real_discriminator_sd),
                                  color='firebrick', alpha=.2)

    ax_discriminator.plot(all_x, fake_discriminator_means, color='salmon', label='Generated words')
    ax_discriminator.fill_between(all_x, (fake_discriminator_means - fake_discriminator_sd), (fake_discriminator_means + fake_discriminator_sd),
                                  color='salmon', alpha=.2)

    total_discriminator_mean = real_discriminator_means + fake_discriminator_means
    total_discriminator_sd = np.sqrt(real_discriminator_sd ** 2 + fake_discriminator_sd ** 2)
    ax_discriminator.plot(all_x, total_discriminator_mean, color='orangered', label='Total Error')
    ax_discriminator.fill_between(all_x, (total_discriminator_mean - total_discriminator_sd), (total_discriminator_mean + total_discriminator_sd),
                                  color='salmon', alpha=.2)

    ax_discriminator.set_xlabel('Epoch')
    ax_discriminator.set_ylabel('Error')
    ax_discriminator.set_title('Discriminator Error - Real and Generated')

    ax_discriminator.legend()
    fig_discriminator.savefig(os.path.join(args.output_plot_dir, 'discriminator_error.png'), bbox_inches='tight')


def main():
    class Args(object):
        def __init__(self):
            # self.logging_filenames_to_plot = ['logging_g-bert-reinforce_c-bert-reinforce-18',
            #                                   'logging_g-bert-reinforce_c-bert-reinforce-16',
            #                                   'logging_g-bert-reinforce_c-bert-reinforce-21',
            #                                   'logging_g-bert-reinforce_c-bert-reinforce-20',
            #                                   'logging_g-bert-reinforce_c-bert-reinforce-17']
            self.logging_filenames_to_plot = ['logging_g-bert-reinforce_c-bert-reinforce-23']
            self.output_dir = 'output/'
            self.logging_dir = '/home/kinne174/private/PythonProjects/gan_qa/logging/all/'

    args = Args()

    args.output_data_dir = os.path.join(args.output_dir, 'all_data/')
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    for lf in args.logging_filenames_to_plot:
        args.current_output_data_dir = os.path.join(args.output_data_dir, lf)
        if not os.path.exists(args.current_output_data_dir) or True:
            gather_data_from_logging(args, os.path.join(args.logging_dir, lf))

    args.output_plot_dir = os.path.join(args.output_dir, 'all_plots/')
    if not os.path.exists(args.output_plot_dir):
        os.makedirs(args.output_plot_dir)

    args.output_plot_dir = os.path.join(args.output_plot_dir, 'bert23/')
    if not os.path.exists(args.output_plot_dir):
        os.makedirs(args.output_plot_dir)
    # else:
    #     raise Exception('This directory already exists!')

    plot_training_errors(args)


if __name__ == '__main__':
    main()