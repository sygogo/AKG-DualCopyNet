import logging
import os
import time
import torch

from src.utils.constants import SEP_WORD, EOS_WORD, UNK_WORD
from src.utils.functions import time_since
from src.utils.string_helper import split_word_list_by_delimiter, prediction_to_sentence


def evaluate_greedy_generator(data_loader, generator, opt):
    pred_output_file = open(os.path.join(opt.pred_path, "predictions_{}.txt".format(opt.model_name)), "w")
    interval = 10
    with torch.no_grad():
        word2idx = opt.vocab['word2idx']
        idx2word = opt.vocab['idx2word']
        start_time = time.time()
        for batch_i, batch in enumerate(data_loader):
            if (batch_i + 1) % interval == 0:
                logging.info("Batch %d: Time for running beam search on %d batches : %.1f" % (
                    batch_i + 1, interval, time_since(start_time)))
                start_time = time.time()

            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str, absent_trg_str, present_trg_str, trg, trg_oov, trg_lens, trg_mask, tags, original_idx_list = batch

            src = src.cuda()
            src_lens = torch.tensor(src_lens)
            src_oov = src_oov.cuda()
            src_mask = src_mask.cuda()

            n_best_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, word2idx, opt)
            pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists,
                                                 word2idx[EOS_WORD],
                                                 word2idx[UNK_WORD],
                                                 opt.replace_unk, src_str_list)

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov, absent_trg_list, present_trg_list in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists, absent_trg_str, present_trg_str):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                # all_keyphrase_list: a list of word list contains all the keyphrases \
                # in the top max_n sequences decoded by beam search
                all_keyphrase_list = []
                for word_list in pred:
                    all_keyphrase_list += split_word_list_by_delimiter(word_list, SEP_WORD)

                # output the predicted keyphrases to a file
                write_example_kp(pred_output_file, all_keyphrase_list)

        pred_output_file.close()


def write_example_kp(out_file, kp_list):
    pred_print_out = ''
    for word_list_i, word_list in enumerate(kp_list):
        if word_list_i < len(kp_list) - 1:
            pred_print_out += '%s;' % ' '.join(word_list)
        else:
            pred_print_out += '%s' % ' '.join(word_list)
    pred_print_out += '\n'
    out_file.write(pred_print_out)


def preprocess_n_best_result(n_best_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk,
                             src_str_list):
    predictions = n_best_result['predictions']
    attention = n_best_result['attention']
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, attn_n_best, oov, src_word_list in zip(predictions, attention, oov_lists, src_str_list):
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            sentences_n_best.append(sentence)
        # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_list.append(sentences_n_best)
    return pred_list
