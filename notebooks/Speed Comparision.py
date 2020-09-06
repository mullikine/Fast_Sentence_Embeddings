#!/usr/bin/env python
# coding: utf-8

# # Speed comparision

from fse.models import Average
from fse.models.average import train_average_np
from fse.models.average_inner import train_average_cy

from fse.models.average import MAX_WORDS_IN_BATCH

from fse import IndexedList

import numpy as np

import gensim.downloader as api
data = api.load("quora-duplicate-questions")

sentences = []
batch_size = 0
for d in data:
    strings = d["question1"].split()
    if len(strings) + batch_size < MAX_WORDS_IN_BATCH:
        sentences.append(strings)
        batch_size += len(strings)
sentences = IndexedList(sentences)

# # Test W2V Model

import gensim.downloader as api

w2v = api.load("glove-wiki-gigaword-100")
ft = api.load("fasttext-wiki-news-subwords-300")

# To test if the fast version is available, you need to import the variable FAST_VERSION from fse.models.average.
# 1 : The cython version is available
# -1 : The cython version is not available.
#
# If the cython compiliation fails, you will be notified.

from fse.models.average import FAST_VERSION
FAST_VERSION

get_ipython().run_cell_magic('timeit', '', 'w2v_avg = Average(w2v)')

get_ipython().run_cell_magic('timeit', '', 'w2v_avg = Average(w2v, lang_freq="en")')

# The slowest part during the init is the induction of frequencies for words, as some pre-trained embeddings do not come with frequencies for words. This is only necessary for the SIF and uSIF Model, not for the Average model.

w2v_avg = Average(w2v)
statistics = w2v_avg.scan_sentences(sentences)
w2v_avg.prep.prepare_vectors(sv=w2v_avg.sv, total_sentences=statistics["max_index"], update=False)
memory = w2v_avg._get_thread_working_mem()

get_ipython().run_cell_magic('timeit', '', 'train_average_np(model=w2v_avg, indexed_sentences=sentences, target=w2v_avg.sv.vectors, memory=memory)')

get_ipython().run_cell_magic('timeit', '', 'train_average_cy(model=w2v_avg, indexed_sentences=sentences, target=w2v_avg.sv.vectors, memory=memory)')

# For 90 sentences, the Cython version is about 8-15 faster than the numpy version when using a Word2Vec type model.

out_w2v_np = np.zeros_like(w2v_avg.sv.vectors)
out_w2v_cy = np.zeros_like(w2v_avg.sv.vectors)
train_average_np(model=w2v_avg, indexed_sentences=sentences, target=out_w2v_np, memory=w2v_avg._get_thread_working_mem())
train_average_cy(model=w2v_avg, indexed_sentences=sentences, target=out_w2v_cy, memory=w2v_avg._get_thread_working_mem())

np.allclose(out_w2v_np, out_w2v_cy)

# # Test FastTextModel

ft_avg = Average(ft)
statistics = ft_avg.scan_sentences(sentences)
ft_avg.prep.prepare_vectors(sv=ft_avg.sv, total_sentences=statistics["max_index"], update=False)
memory = ft_avg._get_thread_working_mem()

get_ipython().run_cell_magic('timeit', '', 'train_average_np(model=ft_avg, indexed_sentences=sentences, target=ft_avg.sv.vectors, memory=memory)')

get_ipython().run_cell_magic('timeit', '', 'train_average_cy(model=ft_avg, indexed_sentences=sentences, target=ft_avg.sv.vectors, memory=memory)')

# With a FastText type model, the cython routine is about 5-10 times faster.

out_ft_np = np.zeros_like(ft_avg.sv.vectors)
out_ft_cy = np.zeros_like(ft_avg.sv.vectors)
train_average_np(model=ft_avg, indexed_sentences=sentences, target=out_ft_np, memory=ft_avg._get_thread_working_mem())
train_average_cy(model=ft_avg, indexed_sentences=sentences, target=out_ft_cy, memory=ft_avg._get_thread_working_mem())

np.allclose(out_ft_np, out_ft_cy)

