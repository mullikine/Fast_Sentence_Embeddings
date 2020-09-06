#!/usr/bin/env python
# coding: utf-8

# # fse - Tutorial

# Welcome to fse - fast sentence embeddings. The library is intended to compute sentence embeddings as fast as possible.
# It offers a simple and easy to understand syntax for you to use in your own projects. Before we start with any model, lets have a look at the input types.
# All fse models require an iterable/generator which produces a tuple. The tuple has two fields: words and index. The index is required for the multi-thread processing, as sentences might not be processed sequentially. The index dictates, which row of the corresponding sentence vector matrix the sentence belongs to.

# ## Input handling

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

s = (["Hello", "world"], 0)
print(s[0])
print(s[1])

# The words of the tuple will always consist of a list of strings. Otherwise the train method will raise an Error. However, most input data is available as a list of strings.

# In order to deal with this common input format, fse provides the IndexedList and some variants, which handel all required data operations for you. You can provide multiple lists (or sets) which will all be merged into a single list. This eases work if you have to work with the STS datasets.
#
# The multiple types of indexed lists. Let's go through them one by one:
# - IndexedList: for already pre-splitted sentences
# - **C**IndexedList: for already pre-splitted sentences with a custom index for each sentence
# - SplitIndexedList: for sentences which have not been splitted. Will split the strings
# - Split**C**IndexedList: for sentences which have not been splitted and with a custom index for each sentence
# - **C**SplitIndexedList: for sentences which have not been splitted. Will split the strings. You can provide a custom split function
# - **C**Split*C*IndexedList: for sentences where you want to provide a custom index and a custom split function.
#
# *Note*: These are ordered by speed. Meaning, that IndexedList is the fastest, while **C**Split*C*IndexedList is the slowest variant.

from fse import SplitIndexedList

sentences_a = ["Hello there", "how are you?"]
sentences_b = ["today is a good day", "Lorem ipsum"]

s = SplitIndexedList(sentences_a, sentences_b)
print(len(s))
s[0]

# To save memory, we do not convert the original lists inplace. The conversion will only take place once you call the getitem method. To access the original data, call:

s.items

# If the data is already preprocessed as a list of lists you can just use the IndexedList

from fse import IndexedList

sentences_splitted = ["Hello there".split(), "how are you?".split()]
s = IndexedList(sentences_splitted)
print(len(s))
s[0]

# In case you want to provide your own splitting function, you can pass a callable to the **C**SplitIndexedList class.

from fse import CSplitIndexedList

def split_func(string):
    return string.lower().split()

s = CSplitIndexedList(sentences_a, custom_split=split_func)
print(len(s))
s[0]

# If you want to stream a file from disk (where each line corresponds to a sentence) you can use the IndexedLineDocument.

from fse import IndexedLineDocument
doc = IndexedLineDocument("../fse/test/test_data/test_sentences.txt")

i = 0
for s in doc:
    print(f"{s[1]}\t{s[0]}")
    i += 1
    if i == 4:
        break

# If you are later working with the similarity of sentences, the IndexedLineDocument provides you the option to access each line by its corresponding index. This helps you in determining the similarity of sentences, as the most_similar method would otherwise just return indices.

doc[20]

# # Training a model / Performing inference

# Training a fse model is simple. You only need a pre-trained word embedding model which you use during the initializiation of the fse model you want to use.

import gensim.downloader as api
data = api.load("quora-duplicate-questions")
glove = api.load("glove-wiki-gigaword-100")

sentences = []
for d in data:
    # Let's blow up the data a bit by replicating each sentence.
    for i in range(8):
        sentences.append(d["question1"].split())
        sentences.append(d["question2"].split())
s = IndexedList(sentences)
print(len(s))

# So we have about 6468640 sentences we want to compute the embeddings for. If you import the FAST_VERSION variable as follows you can ensure, that the compiliation of the cython routines worked correctly:

from fse.models.average import FAST_VERSION, MAX_WORDS_IN_BATCH
print(MAX_WORDS_IN_BATCH)
print(FAST_VERSION)
# 1 -> The fast version works

from fse.models import uSIF
model = uSIF(glove, workers=2, lang_freq="en")

model.train(s)

# The models training speed revolves at around 400,000 - 500,000 sentences / seconds. That means we finish the task in about 10 seconds. This is **heavily dependent** on the input processing. If you train ram-to-ram it is naturally faster than any ram-to-disk or disk-to-disk varianty. Similarly, the speed depends on the workers.

# Once the sif model is trained, you can perform additional inferences for unknown sentences. This two step process for new data is required, as computing the principal components for models like SIF and uSIF will require a fair amount of sentences. If you want the vector for a single sentence (which is out of the training vocab), just use:

tmp = ("Hello my friends".split(), 0)
model.infer([tmp])

# ## Querying the model

# In order to query the model or perform similarity computations we can just access the model.sv (sentence vectors) object and use its method. To get a vector for an index, just call

model.sv[0]

# To compute the similarity or distance between two sentence from the training set you can call:

print(model.sv.similarity(0,1).round(3))
print(model.sv.distance(0,1).round(3))

# We can further call for the most similar sentences given an index. For example, we want to know the most similar sentences for sentence index 100:

print(s[100])

model.sv.most_similar(100)
# Division by zero can happen if you encounter empy sentences

# However, the preceding function will only supply the indices of the most similar sentences. You can circumvent this problem by passing an indexable function to the most_similar call:

model.sv.most_similar(100, indexable=s.items)

# There we go. This is a lot more understandable than the initial list of indices.

# To search for sentences, which are similar to a given word vector, you can call:

model.sv.similar_by_word("easy", wv=glove, indexable=s.items)

# Furthermore, you can query for unknown (or new) sentences by calling:

model.sv.similar_by_sentence("Is this really easy to learn".split(), model=model, indexable=s.items)

# Feel free to browse through the library and get to know the functions a little better!


