import logging
import os
import sys


def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


def vocab_opts(parser):
    # Dictionary Options
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help="Size of the source vocabulary")
    # for copy mechanism
    parser.add_argument('-max_unk_words', type=int, default=1000,
                        help="Maximum number of unknown words the model supports (mainly for masking in loss)")


def predict_opts(parser):
    # beam search
    parser.add_argument('-remove_title_eos', action="store_true", help='Remove the eos after the title')
    parser.add_argument('-beam_size', type=int, default=200,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=None,
                        help='Pick the top n_best sequences from beam_search, if n_best is None, then n_best=beam_size')
    parser.add_argument('-max_length', type=int, default=6,
                        help='Maximum prediction length.')

    parser.add_argument('-length_penalty_factor', type=float, default=0.,
                        help="""Google NMT length penalty parameter
                               (higher = longer generation)""")
    parser.add_argument('-coverage_penalty_factor', type=float, default=0.,
                        help="""Coverage penalty parameter""")
    parser.add_argument('-length_penalty', default='none', choices=['none', 'wu', 'avg'],
                        help="""Length Penalty to use.""")
    parser.add_argument('-coverage_penalty', default='none', choices=['none', 'wu', 'summary'],
                        help="""Coverage Penalty to use.""")
    parser.add_argument('-block_ngram_repeat', type=int, default=0,
                        help='Block repeat of n-gram')
    parser.add_argument('-ignore_when_blocking', nargs='+', type=str,
                        default=['<sep>'],
                        help="""Ignore these strings when blocking repeats.
                                          You want to block sentence delimiters.""")
    # convert index to word options
    parser.add_argument('-replace_unk', action="store_true",
                        help='Replace the unk token with the token of highest attention score.')

    parser.add_argument('-model_path', type=str, default="model",
                        help="Path of checkpoints.")
    parser.add_argument('-model_name', type=str, default="lstm+crf",
                        help="Path of checkpoints.")

    parser.add_argument('-pred_path', type=str, default="pred/%s.%s",
                        help="Path of outputs of predictions.")
    parser.add_argument('-exp_path', type=str, default="exp/%s.%s",
                        help="Path of experiment log/plot.")
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the "vocab.pt"
                            file path from preprocess.py""")

    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=0,
                        help='Number of workers for generating batches')

    parser.add_argument('-src_file', required=True, help="""Path to source file""")

    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                                reproducibility.""")

    parser.add_argument("-local_rank", type=int, default=0, help="GPU ID")


def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data', required=True,
                        help="""Path prefix to the "train.one2one.pt" and
                        "train.one2many.pt" file path from preprocess.py""")
    parser.add_argument('-vocab', required=True,
                        help="""Path prefix to the "vocab.pt"
                        file path from preprocess.py""")
    parser.add_argument('-exp_path', type=str, default="exp",
                        help="Path of experiment log/plot.")
    parser.add_argument('-model_path', type=str, default="model",
                        help="Path of checkpoints.")
    parser.add_argument('-model_name', type=str, default="lstm+crf",
                        help="Path of checkpoints.")

    parser.add_argument('-early_stop_tolerance', type=int, default=3,
                        help="Stop training if it doesn't improve any more for several rounds of validation")

    # Init options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    # parser.add_argument('-weight', type=float, default=0.8,
    #                     help='joint weight')
    parser.add_argument('-batch_workers', type=int, default=0,
                        help='Number of workers for generating batches')

    # Optimization options
    parser.add_argument('-epochs', type=int, default=30,
                        help='Number of training epochs')

    parser.add_argument("-start_epoch_at", type=int, default=0)

    parser.add_argument("-start_validation_epoch", type=int, default=0)

    parser.add_argument('-max_grad_norm', type=float, default=1,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.0001,
                        help="""Starting learning rate.
                            Recommended settings: sgd = 1, adagrad = 0.1,
                            adadelta = 1, adam = 0.001""")

    parser.add_argument("-local_rank", type=int, default=0, help="Gpu id")


def post_predict_opts(parser):
    parser.add_argument('-model_name', type=str, default="lstm+crf",
                        help="Path of checkpoints.")
    parser.add_argument('-pred_file_path', type=str, required=True,
                        help="Path of the prediction file.")
    parser.add_argument('-src_file_path', type=str, required=True,
                        help="Path of the source text file.")
    parser.add_argument('-trg_file_path', type=str,
                        help="Path of the target text file.")
    parser.add_argument('-export_filtered_pred', action="store_true",
                        help="Export the filtered predictions to a file or not")
    parser.add_argument('-filtered_pred_path', type=str, default="",
                        help="Path of the folder for storing the filtered prediction")
    parser.add_argument('-exp_path', type=str, default="",
                        help="Path of experiment log/plot.")
    parser.add_argument('-disable_extra_one_word_filter', action="store_true",
                        help="If False, it will only keep the first one-word prediction")
    parser.add_argument('-disable_valid_filter', action="store_true",
                        help="If False, it will remove all the invalid predictions")
    parser.add_argument('-num_preds', type=int, default=200,
                        help='It will only consider the first num_preds keyphrases in each line of the prediction file')
    parser.add_argument('-debug', action="store_true", default=False,
                        help='Print out the metric at each step or not')
    parser.add_argument('-match_by_str', action="store_true", default=False,
                        help='If false, match the words at word level when checking present keyphrase. Else, match the words at string level.')
    parser.add_argument('-invalidate_unk', action="store_true", default=False,
                        help='Treat unk as invalid output')
    parser.add_argument('-target_separated', action="store_true", default=False,
                        help='The targets has already been separated into present keyphrases and absent keyphrases')
    parser.add_argument('-prediction_separated', action="store_true", default=False,
                        help='The predictions has already been separated into present keyphrases and absent keyphrases')
    parser.add_argument('-reverse_sorting', action="store_true", default=False,
                        help='Only effective in target separated.')
    parser.add_argument('-tune_f1_v', action="store_true", default=False,
                        help='For tuning the F1@V score.')
    parser.add_argument('-all_ks', nargs='+', default=['5', '10', 'M'], type=str,
                        help='only allow integer or M')
    parser.add_argument('-present_ks', nargs='+', default=['5', '10', 'M'], type=str,
                        help='')
    parser.add_argument('-absent_ks', nargs='+', default=['5', '10', '50', 'M'], type=str,
                        help='')
    parser.add_argument('-target_already_stemmed', action="store_true", default=False,
                        help='If it is true, it will not stem the target keyphrases.')
    parser.add_argument('-meng_rui_precision', action="store_true", default=False,
                        help='If it is true, when computing precision, it will divided by the number pf predictions, instead of divided by k.')
    parser.add_argument('-use_name_variations', action="store_true", default=False,
                        help='Match the ground-truth with name variations.')


def preprocess_opts(parser):
    parser.add_argument('-data_dir', required=True, help='The source file of the data')
    parser.add_argument('-save_data_dir', required=True, help='The saving path for the data')
    parser.add_argument('-remove_title_eos', action="store_true", help='Remove the eos after the title')
    parser.add_argument('-log_path', type=str, default="logs")


def lstm_crf_opts(opt):
    opt.vocab_size = 50000
    opt.embedding_dim = 100
    opt.hidden_dim = 150


def cat_seq_opts(opt):
    opt.vocab_size = 50000
    opt.embedding_dim = 100
    opt.hidden_dim = 150
    opt.memory_bank_size = 300
    opt.dropout = 0.1
