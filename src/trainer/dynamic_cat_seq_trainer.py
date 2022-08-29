from nltk import PorterStemmer

from src.inference.evaluate import evaluate_greedy_generator
from src.inference.sequence_generator import SequenceGenerator
from src.trainer.base_trainer import BaseTrainer
import torch
import numpy as np
from src.utils.constants import Tag2Idx, EOS_WORD, SEP_WORD, BOS_WORD, PAD_WORD
from src.utils.metrics import compute_precision, compute_recall

stemmer = PorterStemmer()


class DynamicCatSeqTrainer(BaseTrainer):
    def __init__(self):
        super(DynamicCatSeqTrainer, self).__init__()

    def train_one_batch(self, model, batch, opt):
        src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str, absent_trg_str, present_trg_str, trg, trg_oov, trg_lens, trg_mask, tags, _ = batch
        src = src.cuda()
        src_lens = torch.tensor(src_lens)
        trg = trg.cuda()
        src_oov = src_oov.cuda()
        max_num_oov = max([len(oov) for oov in oov_lists])
        src_mask = src_mask.cuda()
        trg_oov = trg_oov.cuda()
        trg_mask = trg_mask.cuda()
        trg_lens = torch.tensor(trg_lens)
        tags = tags.cuda()
        loss = model(src, src_lens, src_oov, max_num_oov, src_mask, trg, trg_oov, trg_mask, trg_lens, tags)
        # loss = model.compute_loss(decoder_dist_all, trg_oov, trg_mask, trg_lens, memory_bank_ner, tags, ner_mask)
        return loss

    def predict(self, model, test_data_loader, opt):
        generator = SequenceGenerator(model, opt.vocab['word2idx'][EOS_WORD], opt.vocab['word2idx'][BOS_WORD], opt.vocab['word2idx'][PAD_WORD], opt.beam_size, opt.max_length)
        evaluate_greedy_generator(test_data_loader, generator, opt)
