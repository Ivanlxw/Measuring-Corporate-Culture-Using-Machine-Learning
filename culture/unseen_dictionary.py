import pandas as pd
from collections import Counter, OrderedDict
import gensim
import pickle
import itertools
import numpy as np
from multiprocessing import Pool
from functools import partial
from itertools import repeat
import os
import tqdm
from operator import itemgetter
import statistics as s
import math
from collections import defaultdict
from sklearn import preprocessing

def expand_words_dimension_mean(
    word2vec_model,
    seed_words,
    n=50,
    restrict=None,
    min_similarity=0,
    filter_word_set=None,
):
    vocab_number = len(word2vec_model.wv.vocab)
    expanded_words = {}
    all_seeds = set()
    for dim in seed_words.keys():
        all_seeds.update(seed_words[dim])
    if restrict != None:
        restrict = int(vocab_number * restrict)

    for dimension in seed_words:
        dimension_words = [
            word for word in seed_words[dimension] ## if word in word2vec_model.wv.vocab
        ]

        if len(dimension) > 0:
            ## find most similar for each dimension_word that may be unseen
            similar_words = [
                pair[0]
                for pair in word2vec_model.most_similar(
                    dimension_words, topn=n, restrict_vocab=restrict
                )
                if pair[1] >= min_similarity and pair[0] not in all_seeds
            ]
        else:
            similar_words = []

        if filter_word_set is not None:
            similar_words = [x for x in similar_words if x not in filter_word_set]

        similar_words = [
            x for x in similar_words if "[ner:" not in x
        ]  # filter out NERs

        # similar_words = list(map(lambda x: x.replace('_', " "), similar_words))
        expanded_words[dimension] = similar_words

    for dim in expanded_words.keys():
        expanded_words[dim] = expanded_words[dim] + seed_words[dim]

    for d, i in expanded_words.items():
        expanded_words[d] = set(i)

    return expanded_words

def deduplicate_keywords(word2vec_model, expanded_words, seed_words):
    """
    If a word cross-loads, choose the most similar dimension. Return a deduplicated dict. 
    """
    from collections import Counter

    word_counter = Counter()

    for dimension in expanded_words:
        word_counter.update(list(expanded_words[dimension]))
    # for dimension in seed_words:
    #     for w in seed_words[dimension]:
    #         if w not in word2vec_model.wv.vocab:
    #             seed_words[dimension].remove(w)

    word_counter = {k: v for k, v in word_counter.items() if v > 1}  # duplicated words
    dup_words = set(word_counter.keys())
    for dimension in expanded_words:
        expanded_words[dimension] = expanded_words[dimension].difference(dup_words)

    for word in list(dup_words):
        sim_w_dim = {}
        for dimension in expanded_words:
            dimension_seed_words = [
                word
                for word in seed_words[dimension]
                # if word in word2vec_model.wv.vocab
            ]
            # sim_w_dim[dimension] = max([word2vec_model.wv.n_similarity([word], [x]) for x in seed_words[dimension]] )

            sim_w_dim[dimension] = word2vec_model.wv.n_similarity(
                dimension_seed_words, [word]
            )
        max_dim = max(sim_w_dim, key=sim_w_dim.get)
        expanded_words[max_dim].add(word)

    for dimension in expanded_words:
        expanded_words[dimension] = sorted(expanded_words[dimension])

    return expanded_words

def rank_by_sim(expanded_words, seed_words, model, limit=None) -> "dict[str: list]":
    """ Rank each dim in a dictionary based on similarity to the seend words mean
    Returns: expanded_words_sorted {dict[str:list]}
    """
    expanded_words_sorted = dict()
    for dimension in expanded_words.keys():
        dimension_seed_words = [
            word for word in seed_words[dimension] #if word in model.wv.vocab
        ]

        similarity_dict = dict()
        for w in expanded_words[dimension]:
            if w in model.wv.vocab:
                similarity_dict[w] = model.wv.n_similarity(dimension_seed_words, [w])

        sorted_similarity_dict = sorted(
            similarity_dict.items(), key=itemgetter(1), reverse=True
        )

        sorted_similarity_list = [x[0] for x in sorted_similarity_dict]
        if limit == None:
            expanded_words_sorted[dimension] = sorted_similarity_list
        else:
            expanded_words_sorted[dimension] = sorted_similarity_list[:limit]
            
    return expanded_words_sorted