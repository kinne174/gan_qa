import torch
import logging
import os

# logging
logger = logging.getLogger(__name__)


def ablation(args, tokenizer, fake_inputs, inputs, checkpoint, subset, real_predictions, fake_predictions):
    # print question and answers
    # for each answer print the windowed true and fake context and attention scores
    # can try to translate but not necessary on first pass

    assert fake_inputs['input_ids'].shape[1] == inputs['input_ids'].shape[1] == 4, 'One of fake_inputs 2nd dimension ({}) or inputs 2nd dimension ({}) is not 4'.format(fake_inputs['input_ids'].shape[1], inputs['input_ids'].shape[1])

    ablation_dir = os.path.join(args.output_dir, 'ablation_{}'.format(subset))

    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    ablation_filename = os.path.join(ablation_dir, 'checkpoint_{}.txt'.format(checkpoint))
    if os.path.exists(ablation_filename):
        write_append_trigger = 'a'
    else:
        write_append_trigger = 'w'

        with open(ablation_filename, write_append_trigger) as af:
            af.write('Ablation Study start:\n')
            af.write('Reported is the question, answer, score for the answers and how the context was changed to fool the classifier.\n')
            af.write('The two scores in parentheses are the score for the real and the fake respectively\n')
            af.write('\n*********************************************************************************\n')

    assert inputs['input_ids'].shape[0] == fake_inputs['input_ids'].shape[0], 'inputs batch size ({}) is not the same as fake_inputs batch size ({})'.format(inputs['input_ids'].shape[0], fake_inputs['input_ids'].shape[0])
    batch_size = inputs['input_ids'].shape[0]

    answer_letters = ['A.', 'B.', 'C.', 'D.']

    for i in range(batch_size):
        input_ids = inputs['input_ids'][i, :, :]
        change_index_list = [input_ids[1, i] == input_ids[2, i] == input_ids[3, i] == input_ids[0, i] for i in
                             range(input_ids.shape[1])]
        if False not in change_index_list:
            continue

        input_ids = input_ids[0, :].tolist()

        change_index = change_index_list.index(False)

        question_ids = input_ids[:change_index]
        question_words = tokenizer.convert_ids_to_tokens(question_ids)

        seq1_end_index = inputs['token_type_mask'][i, 0, :].tolist().index(1)

        all_answer_words = []
        all_changed_words = []
        for j in range(4):
            answer_ids = input_ids[change_index:seq1_end_index]
            answer_words = tokenizer.convert_ids_to_tokens(answer_ids)
            all_answer_words.append(answer_words)

            pad_index = inputs['attention_mask'].tolist().index(0)

            real_ids = input_ids[seq1_end_index:pad_index]
            fake_ids = fake_inputs['input_ids'][seq1_end_index:pad_index]

            real_words = tokenizer.convert_ids_to_tokens(real_ids)
            fake_words = tokenizer.convert_ids_to_tokens(fake_ids)

            window_size = 4
            my_attention_mask = inputs['my_attention_mask'][i, j, :].tolist()

            assert len(real_words) == len(fake_words) == len(my_attention_mask)

            changed_words = []

            for k in range(len(my_attention_mask)):
                if k == 1:
                    windowed_fake_words = fake_words[max(0, k-window_size):k] + ['*'] + fake_words[k] + ['*'] + fake_words[k+1:min(k+window_size, len(fake_words))]
                    windowed_real_words = real_words[max(0, k-window_size):k] + ['*'] + real_words[k] + ['*'] + real_words[k+1:min(k+window_size, len(real_words))]

                    changed_words.append((windowed_fake_words, windowed_real_words))
            all_changed_words.append(changed_words)

        current_real_label = inputs['labels'][i, :]
        current_fake_label = fake_inputs['labels'][i, :]
        correct_real_label = [' ' if lab == 0 else '*r' for lab in current_real_label]
        correct_fake_label = [' ' if lab == 0 else '*f' for lab in current_fake_label]

        current_real_prediction = torch.argmax(real_predictions[i, :], dim=1)
        current_fake_prediction = torch.argmin(fake_predictions[i, :], dim=1)

        real_predicted_label = [' '] * 4
        real_predicted_label[current_real_prediction.item()] = '#r'
        fake_predicted_label = [' ']*4
        fake_predicted_label[current_fake_prediction.item()] = '#f'

        real_softmaxed_scores = [round(ss, 3) for ss in real_predictions[i, :].squeeze().tolist()]
        fake_softmaxed_scores = [round(ss, 3) for ss in fake_predictions[i, :].squeeze().tolist()]

        assert len(all_changed_words) == len(real_predicted_label) == len(correct_real_label) == len(fake_predicted_label) == len(correct_fake_label) == len(real_softmaxed_scores) == len(fake_softmaxed_scores) == len(all_answer_words) == len(answer_letters)
        answer_features = list(map(tuple, zip(all_changed_words, real_predicted_label, correct_real_label, fake_predicted_label, correct_fake_label, real_softmaxed_scores, fake_softmaxed_scores, all_answer_words, answer_letters)))

        with open(ablation_filename, write_append_trigger) as af:
            af.write('{}. {}\n'.format(i+1, ' '.join(question_words)))

            for (acw, rpl, crl, fpl, cfl, rss, fss, aw, al) in answer_features:
                af.write('{} {} {} {} {} {} {}{}\n'.format(crl, rpl, rss, cfl, fpl, fss ,al, ' '.join(aw)))
                for cw in acw:
                    # cw should be a tuple with the fake words in 0 and real words in 1
                    af.write('\treal: {}\n\tfake: {}\n'.format(' '.join(cw[1]), ' '.join(cw[0])))
                af.write('\n')
            af.write('\n')

    return -1
