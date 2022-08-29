import argparse
import os
import torch
import time
from configs.arg_configs import init_logging, vocab_opts, train_opts, predict_opts
from src.io.dataloader import load_data_and_vocab
from src.trainer.cat_seq_trainer import CatSeqTrainer
from src.trainer.dynamic_cat_seq_trainer import DynamicCatSeqTrainer
from src.trainer.lstm_crf_trainer import LSTMCRFTrainer
from src.utils.constants import SEED
from src.utils.functions import set_seed, time_since


def process_opt(opt):
    """

    :param opt:
    :return:
    """
    set_seed(seed=SEED)
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    return opt


def main(opt):
    """
    :param opt:
    """
    if opt.local_rank == -1:
        opt.device = torch.device('cuda')
    else:
        torch.cuda.set_device(opt.local_rank)
        opt.device = torch.device('cuda', opt.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    if opt.local_rank in [0, -1]:
        start_time = time.time()
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)
    start_time = time.time()
    if opt.model_name == 'lstm_crf':
        trainer = LSTMCRFTrainer()
    elif opt.model_name == 'cat_seq':
        trainer = CatSeqTrainer()
    elif opt.model_name == 'dcat_seq':
        trainer = DynamicCatSeqTrainer()
    train_data_loader, valid_data_loader, vocab = load_data_and_vocab(opt)
    trainer.train_model(train_data_loader, valid_data_loader, opt)

    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    vocab_opts(parser)
    train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    logging = init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    if opt.local_rank in [0, -1]:
        logging.info('Parameters:')
        [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
