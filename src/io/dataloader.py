import logging
import pickle

import torch
from monai.data import worker_init_fn
from torch.utils.data import DataLoader, DistributedSampler

from src.io.keyphrase_dataset import KeyphraseDataset


def load_vocab(opt):
    if opt.local_rank in [0, -1]:
        logging.info("Loading vocab from disk: %s" % opt.vocab)
        # load vocab
    vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')
    # assign vocab to opt
    opt.vocab = vocab
    if opt.local_rank in [0, -1]:
        logging.info('#(vocab)=%d' % len(vocab["word2idx"]))
        logging.info('#(vocab used)=%d' % opt.vocab_size)
    return vocab


def build_data_loader(data, opt, shuffle=False, load_train=True):
    keyphrase_dataset = KeyphraseDataset(examples=data, word2idx=opt.vocab['word2idx'], idx2word=opt.vocab['idx2word'], load_train=load_train)

    if opt.local_rank == -1:
        data_loader = DataLoader(dataset=keyphrase_dataset, collate_fn=keyphrase_dataset.collate_fn_one2seq, num_workers=opt.batch_workers,
                                 batch_size=opt.batch_size, shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        train_sampler = DistributedSampler(keyphrase_dataset)
        data_loader = DataLoader(dataset=keyphrase_dataset, collate_fn=keyphrase_dataset.collate_fn_one2seq, num_workers=opt.batch_workers,
                                 batch_size=opt.batch_size, shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn, sampler=train_sampler)
    return data_loader


def load_data_and_vocab(opt):
    vocab = load_vocab(opt)
    # constructor data loader
    data_path = opt.data + '/%s.one2many.pt'
    # load training dataset

    train_data = torch.load(data_path % "train", 'wb')
    train_loader = build_data_loader(data=train_data, opt=opt, shuffle=False)
    # load validation dataset
    valid_data = torch.load(data_path % "valid", 'wb')
    valid_loader = build_data_loader(data=valid_data, opt=opt, shuffle=False)

    if opt.local_rank in [0, -1]:
        logging.info("Loading train and validate data from '%s'" % opt.data)
        logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))
        logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))

    return train_loader, valid_loader, vocab
