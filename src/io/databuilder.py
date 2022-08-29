import logging
import pickle
from collections import Counter

from src.utils import constants
from src.utils.constants import UNK_WORD
from src.utils.functions import get_tag_label, find_stem_answer
import numpy as np


def build_dataset(src_trgs_pairs, opt, include_original=True):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    word2idx = opt.vocab['word2idx']
    return_examples = []
    oov_target = 0
    max_oov_len = 0
    max_oov_sent = ''

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        # if w's id is larger than opt.vocab_size, replace with <unk>
        src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in source]

        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)

        examples = []  # for one-to-many
        for target in targets:
            example = {}
            if include_original:
                example['src_str'] = source
                example['trg_str'] = target
            example['src'] = src
            example['src_oov'] = src_oov
            example['oov_dict'] = oov_dict
            example['oov_list'] = oov_list
            if len(oov_list) > max_oov_len:
                max_oov_len = len(oov_list)
                max_oov_sent = source

            trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD]
                   for w in target]
            example['trg'] = trg

            # oov words are replaced with new index
            trg_copy = []
            for w in target:
                if w in word2idx and word2idx[w] < opt.vocab_size:
                    trg_copy.append(word2idx[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2idx[UNK_WORD])
            example['trg_copy'] = trg_copy

            if any([w >= opt.vocab_size for w in trg_copy]):
                oov_target += 1

            examples.append(example)

        if len(examples) > 0:
            o2m_example = {}
            keys = examples[0].keys()
            for key in keys:
                if key.startswith('src') or key.startswith('oov') or key.startswith('title'):
                    o2m_example[key] = examples[0][key]
                else:
                    o2m_example[key] = [e[key] for e in examples]

            if include_original:
                assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
            else:
                assert len(o2m_example['src']) == len(o2m_example['src_oov'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])

            result = find_stem_answer(word_list=source, ans_list=targets)
            present_kb = result['present_kp']
            absent_kb = result['absent_kp']
            start_end_pos = result['start_end_pos']
            o2m_example['tags'] = get_tag_label(start_end_pos, len(o2m_example['src_str']))['label']
            o2m_example['absent_trg_str'] = absent_kb
            o2m_example['present_trg_str'] = present_kb

            return_examples.append(o2m_example)

    logging.info('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(return_examples)))
    logging.info('Find max_oov_len = %d' % max_oov_len)
    logging.info('max_oov sentence: %s' % str(max_oov_sent))

    return return_examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    """
    src_oov = []
    oov_dict = {}
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx[UNK_WORD]
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list


def build_interactive_predict_dataset(tokenized_src, opt, include_original=True):
    # build a dummy trg list, and then combine it with src, and pass it to the build_dataset method
    num_lines = len(tokenized_src)
    tokenized_trg = [['.']] * num_lines  # create a dummy tokenized_trg
    tokenized_src_trg_pairs = list(zip(tokenized_src, tokenized_trg))
    return build_dataset(tokenized_src_trg_pairs, opt, include_original=include_original)


def build_vocab(tokenized_src_trg_pairs):
    token_freq_counter = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)

    # Discard special tokens if already present
    special_tokens = [constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD, constants.SEP_WORD, constants.NULL_WORD]
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    vocab = {"word2idx": word2idx, "idx2word": idx2word, "counter": token_freq_counter}
    return vocab
