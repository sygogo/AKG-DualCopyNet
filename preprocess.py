import argparse
import logging
import os
from collections import Counter
import torch

from configs.arg_configs import vocab_opts, preprocess_opts, init_logging
from src.io.databuilder import build_dataset, build_vocab
from src.utils import constants
from src.utils.functions import read_src_and_trg_files


def main(opt):
    # Tokenize train_src and train_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    tokenized_train_pairs = read_src_and_trg_files(opt.train_src, opt.train_trg, is_train=True,
                                                   remove_title_eos=opt.remove_title_eos)
    tokenized_valid_pairs = read_src_and_trg_files(opt.valid_src, opt.valid_trg, is_train=False,
                                                   remove_title_eos=opt.remove_title_eos)

    vocab = build_vocab(tokenized_train_pairs)
    opt.vocab = vocab
    logging.info("Dumping dict to disk: %s" % opt.save_data_dir + '/vocab.pt')

    if not os.path.exists(opt.save_data_dir):
        os.makedirs(opt.save_data_dir)
    torch.save(vocab, open(opt.save_data_dir + '/vocab.pt', 'wb'))

    # saving  one2many datasets
    train_one2many = build_dataset(tokenized_train_pairs, opt)
    logging.info("Dumping train one2many to disk: %s" % (opt.save_data_dir + '/train.one2many.pt'))
    torch.save(train_one2many, open(opt.save_data_dir + '/train.one2many.pt', 'wb'))
    len_train_one2many = len(train_one2many)
    del train_one2many

    valid_one2many = build_dataset(tokenized_valid_pairs, opt)
    logging.info("Dumping valid to disk: %s" % (opt.save_data_dir + '/valid.one2many.pt'))
    torch.save(valid_one2many, open(opt.save_data_dir + '/valid.one2many.pt', 'wb'))

    logging.info('#pairs of train_one2many = %d' % len_train_one2many)
    logging.info('#pairs of valid_one2many = %d' % len(valid_one2many))
    logging.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    vocab_opts(parser)
    preprocess_opts(parser)
    opt = parser.parse_args()
    logging = init_logging(log_file=opt.log_path + "/output.log", stdout=True)
    data_path = os.path.join(opt.save_data_dir, "train.one2many.pt")

    if os.path.exists(data_path):
        logging.info("file exists %s, exit! " % data_path)
        exit()

    opt.train_src = opt.data_dir + '/train_src.txt'
    opt.train_trg = opt.data_dir + '/train_trg.txt'
    opt.valid_src = opt.data_dir + '/valid_src.txt'
    opt.valid_trg = opt.data_dir + '/valid_trg.txt'
    main(opt)
