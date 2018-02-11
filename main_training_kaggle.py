import pandas as pd
import numpy as np

import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import itertools

from tqdm import *
import ast
from math import *
import argparse

import os
from os import listdir
import sys
import math
from time import time
import pickle
from sklearn import preprocessing

import shutil
import warnings
import time
import copy

import nltk 
from collections import Counter

import os.path
from keras.preprocessing.text import Tokenizer


'''
import concurrent.futures
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from main_training_prediction import get_available_gpus
#import tensorflow as tf

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/preprocessing")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/algorithms")

warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
'''


seed = 123
np.random.seed(seed)
#tf.set_random_seed(seed)

model_to_train = {

    0 : {
        'model_file'  : 'feed_forward',
        'model_class' : 'Feed_Forward_Model',
        'layers' : [100],
        'epochs' : 100, 
        'dropout' : [0],
        'batch_size' : 64
    }
}

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_data(tokens):
    
    from nltk.corpus import stopwords
    import string
    from string import maketrans

    table = string.maketrans("","")
    tokens = [w.translate(table, string.punctuation) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]

    return tokens

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_data(data_folder, vocab_counter,is_train):

    directory = data_folder
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue

        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue

        # create the full path of the file to open
        path = directory + '/' + filename
        # load document
        text = load_doc(path)
        tokens = text.split()
        tokens = clean_data(tokens)

        vocab_counter.update(tokens)

        tokens = [w for w in tokens if w in vocab_counter]
        line =  ' '.join(tokens)

        lines.append(line)

    return lines

def generate_bag_of_words(args):
    text = ["The quick brown fox jumped over the lazy dog.",
        "The dog.",
        "The fox"]
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text)
    # summarize
    print(vectorizer.vocabulary_)
    print(vectorizer.idf_)
    # encode document
    vector = vectorizer.transform([text[0]])
    # summarize encoded vector
    print(vector.shape)
    print(vector.toarray())


def train(args):
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.embeddings import Embedding
    # define problem
    vocab_size = 100
    max_length = 32
    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())


    ######

    # define problem
    vocab_size = 100
    max_length = 200
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

def start_threads(args):



    '''with tf.device('/cpu:0'):

        #main_pipeline

        #preprocessor
        from data_preprocessor_logistic_regression import DataPreprocessorLogisticRegression
        data_pre_processor = DataPreprocessorLogisticRegression(args.data_file_path, args.evaluation_file_path, 'target', ['id'], 'remove', args)
        X_train, X_test, y_train, y_test, shuffle_order = data_pre_processor.preprocess()  

        print('Data preprocessed')
        sys.exit()
        gpus_list = get_available_gpus()
    
        for i in range(len(model_to_train)):
            model_to_train[i]['gpu'] = gpus_list.pop()

        
        from logistic_regression import LogisticRegression
        training_algo = LogisticRegression(model_to_train, args, X_train, y_train, X_test, y_test, data_pre_processor)
        
        #output = training_algo.train_random_forest()

        print("K fold")

        output = training_algo.train_random_forest_2()'''

    if os.path.isfile('movie-review-dataset/vocab.txt') == True:

        print("Loading from vocab.txt")

        vocab_filename = 'movie-review-dataset/vocab.txt'
        vocab = load_doc(vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)

        #####

        negative_lines = load_data('movie-review-dataset/txt_sentoken/neg/', vocab, True)
        save_list(negative_lines, 'negative.txt')

        positive_lines = load_data('movie-review-dataset/txt_sentoken/pos/', vocab, True)
        save_list(positive_lines, 'positive.txt')

        # create the tokenizer
        tokenizer = Tokenizer()
        # fit the tokenizer on the documents
        docs = negative_lines + positive_lines
        tokenizer.fit_on_texts(docs)
         
        # encode training data set
        Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
        print(Xtrain.shape)
         
        # load all test reviews
        positive_lines = load_data('movie-review-dataset/txt_sentoken/pos', vocab, False)
        negative_lines = load_data('movie-review-dataset/txt_sentoken/neg', vocab, False)
        docs = negative_lines + positive_lines
        # encode training data set
        Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
        print(Xtest.shape)

    else:

        print("Loading and saving vocab.txt")

        vocab = Counter()
        load_data(args.data_folder, vocab, True)
        load_data('movie-review-dataset/txt_sentoken/pos/',vocab, True)

        min_occurane = 5
        tokens = [k for k,c in vocab.items() if c >= min_occurane]
        print(len(tokens))

        save_list(tokens, 'movie-review-dataset/vocab.txt')

        



    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--seq-len', type=int, default=50, required=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--saving-dir', type=str, default='results/prediction-AbInBev', required=False, help="All results stored in results folder hence results/*new_dirname*")
    parser.add_argument('--data-folder', type=str, default=None, required=True)
    parser.add_argument('--main-root', type=str, default='/home/ubuntu/metis/')
    parser.add_argument('--batch', type=int, default=512)
    #parser.add_argument('--model-file', type=str, required=True,help='Please enter the name of the file which contains the lstm model')
    #parser.add_argument('--model-name', type=str, required=True,help='Please enter the name of the model inside the lstm file')
    parser.add_argument('--tune', type=int, default=0, required=False)

    #parser.add_argument('--data-file-path', type=str, default=None, required=True)
    parser.add_argument('--evaluation-file-path', type=str, default=None, required=False)


    parser.add_argument('--normalization', type=str, default='standard')

    parser.add_argument('--data-start-subset', type=float, default=0.3)
    parser.add_argument('--data-end-subset',   type=float, default=1)

    args = parser.parse_args()

    '''if os.path.isdir(args.main_root + args.saving_dir) == False:
        os.makedirs(args.main_root + args.saving_dir)
    else:
        shutil.rmtree(args.main_root + args.saving_dir)
        os.makedirs(args.main_root + args.saving_dir)'''

    start_threads(args)



if __name__ == "__main__":
    main()
