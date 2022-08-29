import logging
import sys

import torch
from torch.optim import Adam
import torch.nn as nn
from configs.arg_configs import lstm_crf_opts, cat_seq_opts
from src.models.models.cat_seq import CatSeq
from src.models.models.dynamic_cat_seq import DynamicCatSeq
from src.models.models.lstm_crf import BiRnnCrf
from collections import OrderedDict


class BaseTrainer(object):
    def init_model(self, opt, load_from=False):
        logging.info('======================  Model Parameters  =========================')
        if opt.model_name == 'lstm_crf':
            lstm_crf_opts(opt)
            model = BiRnnCrf(opt)
        elif opt.model_name == 'cat_seq':
            cat_seq_opts(opt)
            model = CatSeq(opt)
        elif opt.model_name == 'dcat_seq':
            lstm_crf_opts(opt)
            cat_seq_opts(opt)
            model = DynamicCatSeq(opt)

        if opt.local_rank in [0, -1]:
            logging.info(model)
            total_params = sum([param.nelement() for param in model.parameters()])
            logging.info('model parameters: %d, %.2fM' % (total_params, total_params * 4 / 1024 / 1024))

        if load_from or opt.start_epoch_at > 0:
            new_state_dict = OrderedDict()
            state_dict = torch.load(open('{}/{}.pt'.format(opt.model_path, opt.model_name), 'rb'))
            for k, v in state_dict.items():
                if str(k).startswith('module'):
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

        device = torch.device('cuda')
        opt.device = device
        model = model.to(device)
        # 是否采用分布式
        if opt.local_rank == -1:
            return model
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
            return model

    def init_optimizer(self, model, opt):
        """
        mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
        :param model:
        :param opt:
        :return:
        """
        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                         lr=opt.learning_rate, betas=(0.9, 0.998), eps=1e-09)
        return optimizer

    def eval_model(self, test_data_loader, opt):
        logging.info('======================  Start Evaluation  =========================')
        model = self.init_model(opt, load_from=True)
        self.predict(model, test_data_loader, opt)

    def train_model(self, train_data_loader, valid_data_loader, opt):
        logging.info('======================  Start Training  =========================')
        model = self.init_model(opt)
        optimizer = self.init_optimizer(model, opt)
        model.train()
        best_loss = 1e10
        early_stop_tolerance = 0
        for epoch in range(opt.start_epoch_at, opt.epochs):
            total_batch = 0
            loss_total = 0
            for batch_i, batch in enumerate(train_data_loader):
                total_batch += 1
                optimizer.zero_grad()
                loss = self.train_one_batch(model, batch, opt)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()
                if opt.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            # after each epoch to eval
            if opt.local_rank in [0, -1]:
                model.eval()
                with torch.no_grad():
                    valid_loss_total = 0
                    val_total_batch = 0
                    for val_batch_i, val_batch in enumerate(valid_data_loader):
                        val_total_batch += 1
                        loss = self.train_one_batch(model, val_batch, opt)
                        valid_loss_total += loss.item()
                    valid_loss = valid_loss_total / val_total_batch

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(model.state_dict(), open('{}/{}.pt'.format(opt.model_path, opt.model_name), 'wb'))
                        early_stop_tolerance = 0
                    else:
                        early_stop_tolerance += 1
                        logging.info("Valid loss does not drop,early_stop_tolerance:{}".format(early_stop_tolerance))
                    logging.info("Epoch %d; batch: %d; loss: %.3f; val_loss: %.3f,best_loss: %.3f" %
                                 (epoch, total_batch, loss_total / total_batch, valid_loss, best_loss))
                model.train()
            # stopping
            if early_stop_tolerance >= opt.early_stop_tolerance:
                logging.info("early stopping ...")
                break
                sys.exit()

    def train_one_batch(self, model, batch, opt):
        """

        :param model:
        :param batch:
        :param opt:
        """
        pass

    def predict(self, model, batch, opt):
        """

        :param model:
        :param batch:
        :param opt:
        """
        pass

    def evaluate_prediction(self, pred_list):
        """

        :param pred_list:
        """
        pass
