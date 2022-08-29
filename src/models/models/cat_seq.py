import torch
import torch.nn as nn

from src.models.layers.attention import Attention
from src.models.layers.decoder import Decoder
from src.models.layers.encoder import Encoder
from src.models.loss.masked_loss import masked_cross_entropy
from src.models.loss.masked_softmax import MaskedSoftmax
from src.utils.constants import Tag2Idx, BOS_WORD, EOS_WORD


class CatSeq(nn.Module):
    def __init__(self, opt):
        super(CatSeq, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.opt = opt
        # self.loss = MaskedSoftmax(dim=-1)

    def forward(self, src, src_lens, src_oov, max_num_oov, src_mask, trg, trg_oov, trg_mask, trg_lens):
        batch_size, max_src_len = list(src.size())
        # Encoding
        memory_bank, encoder_final_state = self.encoder(src, src_lens.cpu())
        h_t_init = encoder_final_state.unsqueeze(0)
        max_target_length = trg.size(1)

        decoder_dist_all = []
        # attention_dist_all = []

        y_t_init = trg.new_ones(batch_size) * self.opt.vocab["word2idx"][BOS_WORD]
        for t in range(max_target_length):
            # if t == 0:
            #     pred_counters = trg.new_zeros(batch_size, dtype=torch.uint8)  # [batch_size]
            # else:
            #     re_init_indicators = (y_t_next == self.opt.vocab["word2idx"][EOS_WORD])  # [batch_size]
            #     pred_counters += re_init_indicators

            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next

            decoder_dist, h_t_next, _, attn_dist, p_gen = self.decoder(y_t, h_t, memory_bank, src_mask, max_num_oov, src_oov)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            # attention_dist_all.append(attn_dist.unsqueeze(1))
            y_t_next = trg[:, t]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        # attention_dist_all = torch.cat(attention_dist_all, dim=1)
        loss = self.compute_loss(decoder_dist_all, trg_oov, trg_mask, trg_lens)
        return loss

    def compute_loss(self, decoder_dist, trg_oov, trg_mask, trg_lens):
        return masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens)
