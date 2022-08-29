import logging
import os
from datetime import time

from src.inference.evaluate import write_example_kp
from src.trainer.base_trainer import BaseTrainer
import torch
import numpy as np
from src.utils.constants import Tag2Idx
from src.utils.functions import time_since
from src.utils.metrics import compute_precision, compute_recall


class LSTMCRFTrainer(BaseTrainer):
    def __init__(self):
        super(LSTMCRFTrainer, self).__init__()

    def train_one_batch(self, model, batch, opt):
        src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str, absent_trg_str, present_trg_str, trg, trg_oov, trg_lens, trg_mask, tags, _ = batch
        src = src.cuda()
        tags = tags.cuda()
        loss = model(src, tags)
        return loss

    def predict(self, model, batch, opt):
        self.evaluate_greedy_generator(model, batch, opt)

    def evaluate_greedy_generator(self, model, test_data_loader, opt):
        pred_output_file = open(os.path.join(opt.pred_path, "predictions_{}.txt".format(opt.model_name)), "w")
        pred_data = []
        for batch in test_data_loader:
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str, absent_trg_str, present_trg_str, trg, trg_oov, trg_lens, trg_mask, tags, order = batch
            _, seq_tags, _, _ = model(src.cuda())
            recover_seq = sorted(zip(order, seq_tags, src_str_list, trg_str, absent_trg_str, present_trg_str), key=lambda p: p[0])
            oreder, seq_tags, src_str_list, trg_str, absent_trg_str, present_trg_str = zip(*recover_seq)
            pred_data.append([seq_tags, src_str_list, trg_str, absent_trg_str, present_trg_str])

        for line in pred_data:
            seq_tags, src_str_list, trg_str, absent_trg_str, present_trg_str = line
            for seq, src, trg, absent_trg, present_trg in zip(seq_tags, src_str_list, trg_str, absent_trg_str, present_trg_str):
                pred_kp = get_keyphrases(seq, src)
                write_example_kp(pred_output_file, pred_kp)

        pred_output_file.close()


def get_keyphrases(seq_tags, seq_str):
    keyphrase_start = False
    keyphrase_list = []
    for index, tag in enumerate(seq_tags):
        if tag == Tag2Idx['S']:  # single keyphrase
            one_words = []
            word = seq_str[index]
            one_words.append(word)
            keyphrase_list.append(one_words)
            continue
        if tag == Tag2Idx['B']:
            one_words = []
            keyphrase_start = True
            word = seq_str[index]
            one_words.append(word)
            continue
        if keyphrase_start:
            if tag == Tag2Idx['I']:
                word = seq_str[index]
                one_words.append(word)
            elif tag == Tag2Idx['E']:
                word = seq_str[index]
                one_words.append(word)
                keyphrase_start = False
                keyphrase_list.append(one_words)
                one_words = []
                continue
            elif tag == 0:
                keyphrase_start = False
                keyphrase_list.append(one_words)
    return keyphrase_list
