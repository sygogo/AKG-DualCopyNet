import torch

from src.utils.constants import PAD_WORD, EOS_WORD, SEP_WORD, PEOS_WORD
import numpy as np


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, load_train=True):
        super(KeyphraseDataset, self).__init__()
        keys = ['src', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'absent_trg_str', 'present_trg_str', 'trg', 'trg_copy', 'tags']

        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.load_train = load_train

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.word2idx[PAD_WORD] * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.word2idx[PAD_WORD]).type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def collate_fn_one2seq(self, batches):
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b['trg_copy'])):
                    if trg_idx == trg_size - 1:
                        trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                    else:
                        # trg_concat = [target_1] + [sep] + [target_2] + [sep] + ...
                        trg_concat += trg_phase + [self.word2idx[SEP_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[SEP_WORD]]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)

            return self.collate_fn_common(batches, trg, trg_oov)
        else:
            return self.collate_fn_common(batches)

    def collate_fn_common(self, batches, trg=None, trg_oov=None):
        # source with oov words replaced by <unk>
        src = [b['src'] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [b['src_oov'] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]

        # b['src_str'] is a word_list for source text
        # b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]
        absent_trg_str = [b['absent_trg_str'] for b in batches]
        present_trg_str = [b['present_trg_str'] for b in batches]
        tags = [b['tags'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        src_lens = [len(i) for i in src]
        batch_size = len(src)
        original_indices = list(range(batch_size))

        seq_pairs = sorted(zip(src_lens, src, src_oov, oov_lists, src_str, trg_str, tags, absent_trg_str, present_trg_str, original_indices), key=lambda p: p[0], reverse=True)
        _, src, src_oov, oov_lists, src_str, trg_str, tags, absent_trg_str, present_trg_str, original_indices = zip(*seq_pairs)

        if self.load_train:
            seq_pairs = sorted(zip(src_lens, trg, trg_oov), key=lambda p: p[0], reverse=True)
            _, trg, trg_oov = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)
        tags, _, _ = self._pad(tags)

        # src = src.cuda()
        # src_mask = src_mask.cuda()
        # src_oov = src_oov.cuda()
        # tags = tags.cuda()

        if self.load_train:
            trg, trg_lens, trg_mask = self._pad(trg)
            trg_oov, _, _ = self._pad(trg_oov)

            # trg = trg.cuda()
            # trg_mask = trg_mask.cuda()
            # trg_oov = trg_oov.cuda()
        else:
            trg_lens, trg_mask = None, None

        return src, src_lens, src_mask, src_oov, oov_lists, src_str, trg_str, absent_trg_str, present_trg_str, trg, trg_oov, trg_lens, trg_mask, tags, original_indices
