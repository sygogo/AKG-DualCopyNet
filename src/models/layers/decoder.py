import torch
import torch.nn as nn

from src.models.layers.attention import Attention
from src.models.loss.masked_softmax import MaskedSoftmax


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
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
        p_gen_input_size = self.embed_size + self.hidden_size*2 + self.memory_bank_size
        self.p_gen_linear = nn.Linear(p_gen_input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.vocab_dist_linear_1 = nn.Linear(self.hidden_size * 2 + self.memory_bank_size, self.hidden_size * 2)
        self.vocab_dist_linear_2 = nn.Linear(self.hidden_size * 2, self.vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, h, memory_bank, src_mask, max_num_oovs, src_oov):
        batch_size, max_src_seq_len = list(src_oov.size())
        y_emb = self.embedding(y).unsqueeze(0)
        rnn_input = y_emb
        _, h_next = self.rnn(rnn_input, h)
        last_layer_h_next = h_next[-1, :, :]
        context, attn_dist = self.attention(last_layer_h_next, memory_bank, src_mask)
        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)
        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))
        p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
        p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        if max_num_oovs > 0:
            # extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
            extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
            vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

        final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
        return final_dist, h_next, context, attn_dist, p_gen
