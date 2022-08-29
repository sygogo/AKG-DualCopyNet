import torch
import torch.nn as nn

from src.models.layers.attention import Attention
from src.models.loss.masked_softmax import MaskedSoftmax
from src.utils.constants import PAD_WORD
import numpy as np


class NERDecoder(nn.Module):
    def __init__(self, opt):
        super(NERDecoder, self).__init__()
        self.embed_size = opt.embedding_dim
        self.hidden_size = opt.hidden_dim
        self.vocab_size = opt.vocab_size
        self.memory_bank_size = opt.memory_bank_size
        self.dropout = nn.Dropout(opt.dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size
        )
        self.rnn = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size * 2, bidirectional=False, dropout=opt.dropout)
        self.attention = Attention(decoder_size=self.hidden_size * 2, memory_bank_size=self.memory_bank_size, attn_mode='concat')
        self.attention1 = Attention(decoder_size=self.hidden_size * 2, memory_bank_size=self.memory_bank_size, attn_mode='concat')
        p_gen_input_size = self.embed_size + self.hidden_size * 2 + self.memory_bank_size
        self.p_gen_linear = nn.Linear(p_gen_input_size, 1)
        self.p_gen_linear1 = nn.Linear(p_gen_input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.vocab_dist_linear_1 = nn.Linear(self.hidden_size * 2 + self.memory_bank_size, self.hidden_size * 2)
        self.vocab_dist_linear_2 = nn.Linear(self.hidden_size * 2, self.vocab_size)
        #
        self.vocab_dist_linear_3 = nn.Linear(self.hidden_size * 2 + self.memory_bank_size, self.hidden_size * 2)
        self.vocab_dist_linear_4 = nn.Linear(self.hidden_size * 2, self.vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, h, memory_bank, src_mask, max_num_oovs, src_oov, ner_seq, ner_seq_memory_bank):
        batch_size, max_src_seq_len = list(src_oov.size())
        y_emb = self.embedding(y).unsqueeze(0)
        rnn_input = y_emb
        _, h_next = self.rnn(rnn_input, h)
        last_layer_h_next = h_next[-1, :, :]
        # oov copy attention
        context, attn_dist = self.attention(last_layer_h_next, memory_bank, src_mask)
        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)
        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))

        p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
        p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist

        # ner copy attention
        context1, attn_dist1 = self.attention1(last_layer_h_next, ner_seq_memory_bank.contiguous(), src_mask)
        vocab_dist_input1 = torch.cat((context1, last_layer_h_next), dim=1)
        vocab_dist1 = self.softmax(self.vocab_dist_linear_4(self.dropout(self.vocab_dist_linear_3(vocab_dist_input1))))

        p_gen_input1 = torch.cat((context1, last_layer_h_next, y_emb.squeeze(0)), dim=1)
        p_gen1 = self.sigmoid(self.p_gen_linear1(p_gen_input1))
        vocab_dist_1 = p_gen1 * vocab_dist1
        attn_dist_1 = (1 - p_gen1) * attn_dist1

        if max_num_oovs > 0:
            # extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
            extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
            vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)
            vocab_dist_1 = torch.cat((vocab_dist_1, extra_zeros), dim=1)

        final_dist1 = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
        final_dist2 = vocab_dist_1.scatter_add(1, src_oov, attn_dist_1)
        final_dist = final_dist1 + final_dist2
        return final_dist, h_next, context, attn_dist, p_gen

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = 0 * np.ones((len(input_list), max_seq_len))
        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]
        padded_batch = torch.LongTensor(padded_batch)
        input_mask = torch.ne(padded_batch, 0).type(torch.FloatTensor)
        return padded_batch, input_list_lens, input_mask
