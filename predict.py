import argparse
import os
import time
import torch

from configs.arg_configs import vocab_opts, init_logging, predict_opts
from src.inference.evaluate import evaluate_greedy_generator
from src.inference.sequence_generator import SequenceGenerator
from src.io.databuilder import build_dataset, build_interactive_predict_dataset
from src.io.dataloader import load_vocab, build_data_loader
from src.trainer.cat_seq_trainer import CatSeqTrainer
from src.trainer.dynamic_cat_seq_trainer import DualCatSeqTrainer
from src.trainer.lstm_crf_trainer import LSTMCRFTrainer
from src.utils.functions import common_process_opt, read_tokenized_src_file, time_since


def process_opt(opt):
    opt = common_process_opt(opt)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    return opt


def predict(test_data_loader, model, opt):
    generator = SequenceGenerator(model, opt)
    evaluate_greedy_generator(test_data_loader, generator, opt)


def main(opt):
    vocab = load_vocab(opt)
    src_file = opt.src_file

    tokenized_src = read_tokenized_src_file(src_file, remove_title_eos=opt.remove_title_eos)

    test_data = build_interactive_predict_dataset(tokenized_src, opt, include_original=True)

    torch.save(test_data, open(opt.pred_path + "/test.pt", 'wb'))

    test_loader = build_data_loader(data=test_data, opt=opt, load_train=False)
    logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

    if opt.model_name == 'lstm_crf':
        trainer = LSTMCRFTrainer()
    elif opt.model_name == 'cat_seq':
        trainer = CatSeqTrainer()
    elif opt.model_name == 'dcat_seq':
        trainer = DualCatSeqTrainer()

    # Print out predict path
    logging.info("Prediction path: %s" % opt.pred_path)

    # predict the keyphrases of the src file and output it to opt.pred_path/predictions.txt
    start_time = time.time()
    trainer.eval_model(test_loader, opt)
    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)


if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(description='interactive_predict.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    vocab_opts(parser)
    predict_opts(parser)
    opt = parser.parse_args()

    opt = process_opt(opt)
    logging = init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
