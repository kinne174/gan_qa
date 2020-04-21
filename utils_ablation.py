import torch
import torch.nn as nn
import logging
import os

# logging
logger = logging.getLogger(__name__)

sigmoid = nn.Sigmoid()


def translate_tokens(args, tokens):

    if args.transformer_name == 'roberta':
        current_tokens = []
        for token in tokens:
            if 'Ġ' not in token:
                current_tokens.append(token)
            else:
                current_tokens.append(token[1:])

    elif args.transformer_name == 'albert':
        current_tokens = []
        for token in tokens:
            if '▁' not in token:  # don't know what this character is, had to copy paste from debugger
                current_tokens.append(token)
            else:
                current_tokens.append(token[1:])

    else:
        return NotImplementedError

    return current_tokens



def ablation(args, ablation_filename, tokenizer, fake_inputs, inputs, real_predictions, fake_predictions):
    # print question and answers
    # for each answer print the windowed true and fake context and attention scores
    # can try to translate but not necessary on first pass

    if not args.transformer_name in ['albert', 'roberta']:
        raise NotImplementedError

    assert fake_inputs['input_ids'].shape[1] == inputs['input_ids'].shape[1] == 4, 'One of fake_inputs 2nd dimension ({}) or inputs 2nd dimension ({}) is not 4'.format(fake_inputs['input_ids'].shape[1], inputs['input_ids'].shape[1])

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

        change_index = change_index_list.index(False)

        question_ids = input_ids[0, :][:change_index]
        question_words = tokenizer.convert_ids_to_tokens(question_ids)

        all_answer_words = []
        all_real_words = []
        all_fake_words = []
        for j in range(4):
            seq_end_index = inputs['token_type_ids'][i, j, :].tolist().index(1)
            if args.transformer_name == 'roberta':
                seq_end_index = inputs['token_type_ids'][i, j, 1:].tolist().index(1) + 1

            answer_ids = inputs['input_ids'][i, j, change_index:seq_end_index]
            answer_words = tokenizer.convert_ids_to_tokens(answer_ids)
            all_answer_words.append(answer_words)

            attention_list = inputs['attention_mask'][i, j, :].tolist()
            if 0 in attention_list:
                pad_index = attention_list.index(0)
            else:
                pad_index = len(attention_list) - 1

            real_ids = inputs['input_ids'][i, j, seq_end_index:pad_index]
            fake_ids = fake_inputs['inputs_embeds'].nonzero()
            fake_ids = fake_ids[:, 1].view(*fake_inputs['input_ids'].shape)
            fake_ids = fake_ids[i, j, seq_end_index:pad_index].long()

            real_words = tokenizer.convert_ids_to_tokens(real_ids)
            fake_words = tokenizer.convert_ids_to_tokens(fake_ids)

            my_attention_mask = fake_inputs['my_attention_mask'][i, j, seq_end_index:pad_index].tolist()

            assert len(real_words) == len(fake_words) == len(my_attention_mask), 'Len of real_words ({}), len of fake_words ({}) and len of my_attention_mask ({}) does not match'.format(len(real_words),
                                                                                                                                                                                          len(fake_words),
                                                                                                                                                                                          len(my_attention_mask))

            att_counter = 1
            new_fake_words = []
            new_real_words = []
            for k, att in enumerate(my_attention_mask):
                if att == 1:
                    new_fake_words.extend(['*{}'.format(att_counter)] + [fake_words[k]] + ['*{}'.format(att_counter)])
                    new_real_words.extend(['*{}'.format(att_counter)] + [real_words[k]] + ['*{}'.format(att_counter)])

                    att_counter += 1
                else:
                    new_fake_words.append(fake_words[k])
                    new_real_words.append(real_words[k])

            all_fake_words.append(new_fake_words)
            all_real_words.append(new_real_words)

        current_real_label = inputs['classification_labels'][i, :]
        current_fake_label = fake_inputs['classification_labels'][i, :]
        correct_real_label = ['  ' if lab == 0 else '*r' for lab in current_real_label]
        correct_fake_label = ['  ' if lab == 1 else '*f' for lab in current_fake_label]

        current_real_prediction = torch.argmax(real_predictions[i, :])
        current_fake_prediction = torch.argmin(fake_predictions[i, :])

        real_predicted_label = ['  '] * 4
        real_predicted_label[current_real_prediction.item()] = '#r'
        fake_predicted_label = ['  ']*4
        fake_predicted_label[current_fake_prediction.item()] = '#f'

        real_softmaxed_scores = [round(ss, 3) for ss in sigmoid(real_predictions[i, :]).squeeze().tolist()]
        fake_softmaxed_scores = [round(ss, 3) for ss in sigmoid(fake_predictions[i, :]).squeeze().tolist()]

        assert len(all_real_words) == len(all_fake_words) == len(real_predicted_label) == len(correct_real_label) == len(fake_predicted_label) == len(correct_fake_label) == len(real_softmaxed_scores) == len(fake_softmaxed_scores) == len(all_answer_words) == len(answer_letters)
        answer_features = list(map(tuple, zip(all_real_words, all_fake_words, real_predicted_label, correct_real_label, fake_predicted_label, correct_fake_label, real_softmaxed_scores, fake_softmaxed_scores, all_answer_words, answer_letters)))

        with open(ablation_filename, write_append_trigger) as af:
            af.write('Predicted real answer: #r. Correct real answer: *r.\n')
            af.write('Predicted fake wrong answer: #f. Correct fake wrong answer: *f.\n\n')

            question_words = translate_tokens(args, question_words)
            af.write('** {}\n'.format(' '.join(question_words)))

            for (rw, fw, rpl, crl, fpl, cfl, rss, fss, aw, al) in answer_features:
                aw = translate_tokens(args, aw)
                af.write('{} {} {} {} {} {} {}{}\n'.format(crl, rpl, rss, cfl, fpl, fss, al, ' '.join(aw)))

                rw = translate_tokens(args, rw)
                fw = translate_tokens(args, fw)

                af.write('Real context: {}\n'.format(' '.join(rw)))
                af.write('Fake context: {}\n\n'.format(' '.join(fw)))

            af.write('\n')

    return -1


def ablation_discriminator(args, ablation_filename, tokenizer, fake_inputs, inputs, fake_predictions, real_predictions, generator_predictions):
    if not args.transformer_name in ['albert', 'roberta']:
        raise NotImplementedError

    assert fake_inputs['input_ids'].shape[1] == inputs['input_ids'].shape[1] == 4, 'One of fake_inputs 2nd dimension ({}) or inputs 2nd dimension ({}) is not 4'.format(fake_inputs['input_ids'].shape[1], inputs['input_ids'].shape[1])

    if os.path.exists(ablation_filename):
        write_append_trigger = 'a'
    else:
        write_append_trigger = 'w'

        with open(ablation_filename, write_append_trigger) as af:
            af.write('Ablation Study start:\n\n')

    assert inputs['input_ids'].shape[0] == fake_inputs['input_ids'].shape[0], 'inputs batch size ({}) is not the same as fake_inputs batch size ({})'.format(inputs['input_ids'].shape[0], fake_inputs['input_ids'].shape[0])
    batch_size = inputs['input_ids'].shape[0]

    for i in range(batch_size):

        for j in range(4):
            # seq_end_index = inputs['token_type_ids'][i, j, :].tolist().index(1)

            attention_list = inputs['attention_mask'][i, j, :].tolist()
            if 0 in attention_list:
                pad_index = attention_list.index(0)
            else:
                pad_index = len(attention_list) - 1

            real_ids = inputs['input_ids'][i, j, :pad_index]
            if 'inputs_embeds' in fake_inputs:
                fake_ids = fake_inputs['inputs_embeds'].nonzero()
                fake_ids = fake_ids[:, 1].view(*fake_inputs['input_ids'].shape)
                fake_ids = fake_ids[i, j, :pad_index].long()
            else:
                fake_ids = fake_inputs['input_ids'][i, j, :pad_index]

            real_words = tokenizer.convert_ids_to_tokens(real_ids)
            fake_words = tokenizer.convert_ids_to_tokens(fake_ids)

            my_attention_mask = fake_inputs['my_attention_mask'][i, j, :pad_index].tolist()

            assert len(real_words) == len(fake_words) == len(
                my_attention_mask), 'Len of real_words ({}), len of fake_words ({}) and len of my_attention_mask ({}) does not match'.format(
                len(real_words),
                len(fake_words),
                len(my_attention_mask))

            att_counter = 1
            new_fake_words = []
            new_real_words = []
            for k, att in enumerate(my_attention_mask):
                if att == 1:
                    new_fake_words.append('({}, {:.3f}, {:.3f}) *{}*'.format(''.join(translate_tokens(args, fake_words[k])), fake_predictions[i, j, k].item(), generator_predictions[i, j, k].item(), att_counter))
                    new_real_words.append('*{}* ({}, {:.3f})'.format(att_counter, ''.join(translate_tokens(args, real_words[k])), real_predictions[i, j, k].item()))

                    att_counter += 1
                else:
                    new_fake_words.append('({}, {:.3f}, {:.3f})'.format(''.join(translate_tokens(args, fake_words[k])), fake_predictions[i, j, k].item(), generator_predictions[i, j, k].item()))
                    new_real_words.append('({}, {:.3f})'.format(''.join(translate_tokens(args, real_words[k])), real_predictions[i, j, k].item()))

            with open(ablation_filename, write_append_trigger) as af:

                # af.write('Real score: {}\n'.format(' '.join([str(round(rp.item(), 3)) for rp in real_predictions[i, j, :pad_index]])))
                # af.write('Fake score: {}\n'.format(' '.join([str(round(fp.item(), 3)) for fp in fake_predictions[i, j, :pad_index]])))

                # rw = translate_tokens(args, new_real_words)
                # fw = translate_tokens(args, new_fake_words)

                af.write('(Real Words, #score#), (Generated Words, #discriminator score#, #generator score#)\n')
                for rw, fw in zip(new_real_words, new_fake_words):
                    af.write('{}, {}\n'.format(rw, fw))

                af.write('\n')

