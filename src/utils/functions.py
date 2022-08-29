import time
import random
import unicodedata

import numpy as np
import logging

import torch

from src.utils.constants import Tag2Idx
from nltk import SnowballStemmer, PorterStemmer

# stemmer_spanish = SnowballStemmer("spanish")

stemmer = PorterStemmer()


def common_process_opt(opt):
    if opt.seed > 0:
        set_seed(opt.seed)
    return opt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_since(start_time):
    return time.time() - start_time


def read_tokenized_src_file(path, remove_title_eos=True):
    """
    read tokenized source text file and convert them to list of list of words
    :param path:
    :param remove_title_eos: concatenate the words in title and content
    :return: data, a 2d list, each item in the list is a list of words of a src text, len(data) = num_lines
    """
    tokenized_train_src = []
    for line_idx, src_line in enumerate(open(path, 'r')):
        # process source line
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_title_eos:
                src_word_list = title_word_list + context_word_list
            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
    return tokenized_train_src


def read_tokenized_trg_file(path):
    """
    read tokenized target text file and convert them to list of list of words
    :param path:
    :return: data, a 3d list, each item in the list is a list of target, each target is a list of words.
    """
    data = []
    with open(path) as f:
        for line in f:
            trg_list = line.strip().split(';')  # a list of target sequences
            trg_word_list = [trg.split(' ') for trg in trg_list]
            data.append(trg_word_list)
    return data


def read_src_and_trg_files(src_file, trg_file, is_train, remove_title_eos=True):
    tokenized_train_src = []
    tokenized_train_trg = []
    filtered_cnt = 0
    for line_idx, (src_line, trg_line) in enumerate(zip(open(src_file, 'r'), open(trg_file, 'r'))):
        # process source line
        if (len(src_line.strip()) == 0) and is_train:
            continue
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_title_eos:
                src_word_list = title_word_list + context_word_list
            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        # process target line
        trg_list = trg_line.strip().split(';')  # a list of target sequences
        trg_word_list = [trg.split(' ') for trg in trg_list]
        # If it is training data, ignore the line with source length > 400 or target length > 60
        if is_train:
            if len(src_word_list) > 400 or len(trg_word_list) > 14:
                filtered_cnt += 1
                continue
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
        tokenized_train_trg.append(trg_word_list)

    assert len(tokenized_train_src) == len(
        tokenized_train_trg), 'the number of records in source and target are not the same'

    logging.info("%d rows filtered" % filtered_cnt)

    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))
    return tokenized_train_pairs


def get_tag_label(start_end_pos, doc_length):
    # flatten, rank, filter overlap for answer positions
    sorted_positions = flat_rank_pos(start_end_pos)
    filter_positions = strict_filter_overlap(sorted_positions)

    if len(filter_positions) != len(sorted_positions):
        overlap_flag = True
    else:
        overlap_flag = False

    label = [Tag2Idx["O"]] * doc_length
    for s, e in filter_positions:
        if s == e:
            label[s] = Tag2Idx["S"]

        elif (e - s) == 1:
            label[s] = Tag2Idx["B"]
            label[e] = Tag2Idx["E"]

        elif (e - s) >= 2:
            label[s] = Tag2Idx["B"]
            label[e] = Tag2Idx["E"]
            for i in range(s + 1, e):
                label[i] = Tag2Idx["I"]
        else:
            print("ERROR")
            break
    return {"label": label, "overlap_flag": overlap_flag}


def flat_rank_pos(start_end_pos):
    flatten_postions = [pos for poses in start_end_pos for pos in poses]
    sorted_positions = sorted(flatten_postions, key=lambda x: x[0])
    return sorted_positions


def strict_filter_overlap(positions):
    """delete overlap keyphrase positions. """
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def norm_doc_to_char(word_list):
    norm_char = unicodedata.normalize("NFD", " ".join(word_list))
    stem_char = " ".join([stemmer.stem(w.strip()) for w in norm_char.split(" ")])

    return norm_char, stem_char


def norm_phrase_to_char(phrase_list):
    # del same keyphrase & null phrase
    norm_phrases = set()
    for phrase in phrase_list:
        p = " ".join([w.strip() for w in phrase if len(w.strip()) > 0])
        if len(p) < 1:
            continue
        norm_phrases.add(unicodedata.normalize("NFD", p))

    norm_stem_phrases = []
    for norm_chars in norm_phrases:
        stem_chars = " ".join([stemmer.stem(w) for w in norm_chars.split(" ")])
        norm_stem_phrases.append((norm_chars, stem_chars))

    return norm_stem_phrases


def find_stem_answer(word_list, ans_list):
    norm_doc_char, stem_doc_char = norm_doc_to_char(word_list)
    norm_stem_phrase_list = norm_phrase_to_char(ans_list)

    present_kp_ans_str = []
    absent_kp_ans_str = []
    present_kp_start_end_pos = []

    for norm_ans_char, stem_ans_char in norm_stem_phrase_list:
        norm_stem_doc_char = " ".join([norm_doc_char, stem_doc_char])
        if (
                norm_ans_char not in norm_stem_doc_char
                and stem_ans_char not in norm_stem_doc_char
        ):
            absent_kp_ans_str.append(norm_ans_char.split())
        else:
            norm_doc_words = norm_doc_char.split(" ")
            stem_doc_words = stem_doc_char.split(" ")

            norm_ans_words = norm_ans_char.split(" ")
            stem_ans_words = stem_ans_char.split(" ")

            assert len(norm_doc_words) == len(stem_doc_words)
            assert len(norm_ans_words) == len(stem_ans_words)

            # find postions
            tot_pos = []

            for i in range(0, len(stem_doc_words) - len(stem_ans_words) + 1):

                Flag = False

                if norm_ans_words == norm_doc_words[i: i + len(norm_ans_words)]:
                    Flag = True

                elif stem_ans_words == norm_doc_words[i: i + len(stem_ans_words)]:
                    Flag = True

                elif norm_ans_words == stem_doc_words[i: i + len(norm_ans_words)]:
                    Flag = True

                elif stem_ans_words == stem_doc_words[i: i + len(stem_ans_words)]:
                    Flag = True

                if Flag:
                    tot_pos.append([i, i + len(norm_ans_words) - 1])
                    assert (i + len(stem_ans_words) - 1) >= i

            if len(tot_pos) > 0:
                present_kp_start_end_pos.append(tot_pos)
                present_kp_ans_str.append(norm_ans_char.split())

    assert len(present_kp_ans_str) == len(present_kp_start_end_pos)
    assert len(word_list) == len(norm_doc_char.split(" "))

    return {"present_kp": present_kp_ans_str, "absent_kp": absent_kp_ans_str, "start_end_pos": present_kp_start_end_pos}
