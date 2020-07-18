import gensim 
import global_options
import pandas as pd
import pickle
from collections import OrderedDict, Counter
import itertools
from pprint import pprint
from culture import unseen_dictionary, culture_dictionary
from pathlib import Path

## load skip-gram model
ft_model = gensim.models.fasttext.FastText.load(
    "models/w2v/w2v_ar_all"
)

expanded_words = unseen_dictionary.expand_words_dimension_mean(
    word2vec_model=ft_model,
    seed_words=global_options.SEED_WORDS,
    restrict=global_options.DICT_RESTRICT_VOCAB,
    n=global_options.N_WORDS_DIM,
)

expanded_words = unseen_dictionary.deduplicate_keywords(
    word2vec_model=ft_model,
    expanded_words=expanded_words,
    seed_words=global_options.SEED_WORDS,
)

expanded_words = unseen_dictionary.rank_by_sim(
    expanded_words, global_options.SEED_WORDS, ft_model, limit=100
)

# print(expanded_words["data analytics"])

filename = "expanded_dict_us_ESGAndFintech.csv"
culture_dictionary.write_dict_to_csv(
    culture_dict=expanded_words,
    file_name=str(Path(global_options.OUTPUT_FOLDER, "dict", filename)),
)

print("Dictionary saved at {}".format(str(Path(global_options.OUTPUT_FOLDER, "dict", filename))))

## similarity operations
#  model_wrapper.similarity("night", "nights")  ## similarity
# model_wrapper.most_similar("nights")

## compute distance from manual words
