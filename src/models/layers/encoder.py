import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.vocab_size = opt.vocab_size
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True, dropout=opt.dropout)

    def forward(self, src, src_lens):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        # Debug
        # if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)
        # only extract the final state in the last layer
        encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]), 1)  # [batch, hidden_size*2]

        return memory_bank.contiguous(), encoder_last_layer_final_state
