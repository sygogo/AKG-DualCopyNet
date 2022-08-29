import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.layers.CRF import CRF
from src.utils.constants import Tag2Idx
import torch


class BiRnnCrf(nn.Module):
    """
    LSTM + CRF
    """

    def __init__(self, opt):
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim * 2
        self.vocab_size = opt.vocab_size
        self.tagset_size = len(Tag2Idx)
        self.embedding = nn.Embedding(opt.vocab_size, self.embedding_dim)
        self.crf = CRF(self.hidden_dim, self.tagset_size)
        self.encoder = nn.LSTM(batch_first=True, input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2, bidirectional=True, num_layers=4)

    def __build_features(self, sentences):
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())
        lstm_out, _ = self.encoder(embeds)
        return lstm_out, masks

    def compute_loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, tags=None):
        # Get the emission scores from the BiLSTM
        if tags is None:
            features, masks = self.__build_features(xs)
            scores, tag_seq = self.crf(features, masks)
            return scores, tag_seq, features, masks
        else:
            return self.compute_loss(xs, tags)
