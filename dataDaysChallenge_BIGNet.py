#!/usr/bin/env python
# coding: utf-8

# Programming by Mojtaba Valipour @ SUTest-V1.0.0, vpcom.ir
# Copyright 2019
# Title: Deep Hierarchical Persian Text Classification based on hdlTex

# Information about the environments
# Environment: hdlTex, vpcomDesk -> hdlTex.yml
# Anaconda
# Python:3.5.6
# Tensorflow: 1.10.0
# Keras: 2.2.2
# Pandas: 0.23.4
# nltk: 3.3.0
# numpy: 1.15.2
# Cuda:9.0

# GPU: Geforce GTX 1080

# github.com/mvpcom/ddCh2

# RESOURCES:
'''
1- https://github.com/kk7nc/HDLTex
2- https://nlp.stanford.edu/projects/glove/
3- https://research.cafebazaar.ir/visage/divar_datasets/
4- HDLTex: Hierarchical Deep Learning for Text Classification
5- https://fasttext.cc/docs/en/crawl-vectors.html
6- https://github.com/hadifar/PNLP
7- https://datadays.ir
8- https://github.com/philipperemy/keras-attention-mechanism
'''''' Challenge2_DivarDataset_DataDays, Sharif University
بخش اول
عنوان: پیش بینی دسته بندی
امتیاز: ۳۰۰۰ امتیاز
توانایی: یادگیری ماشین و تحلیل متن
مسئله: پیشبینی دسته بندی آگهی از روی سایر ویژگی های آن

توصیف: در این بخش شما یک دیتاست شامل ۲۰۰ هزار سطر دانلود میکنید که هر سطر حاوی اطلاعات مربوط به یک آگهی است. شما باید دسته بندی سلسله مراتبی هر آگهی را به دست آورید و در قالب یک فایل csv که شامل ۲۰۰ هزار سطر و سه ستون cat1, cat2, cat3 است آپلود کنید.
ملاحظه مهم: ساختار پاسخ باید دقیقا به شکل اشاره شده باشد. ضمنا تمام دسته ها باید به همان شکلی که در دیتاست Train قرار دارد باشد. یک نمونه از پاسخ مطلوب در این فایل فایل پیوست شده است. 
'''

# Loading Libraries 
import os
import gc
import re
import glob
import json
import tabulate
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.layers.core import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from sklearn.utils import class_weight
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import multiply,Add, Dense, Input, Flatten, Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout, LSTM, GRU, Bidirectional,SimpleRNN

#np.set_printoptions(threshold=np.inf)
# get_ipython().run_line_magic('matplotlib', 'inline')

'''
Config: All Categories Acc:  0.9433751213541007  Cat1 Acc:  0.98198683044194  Cat2 Acc:  0.9710966189692288  Cat3 Acc:  0.9520493014224811
epochs = 15;level2Epochs = 25;level3Epochs = 40;MAX_SEQUENCE_LENGTH = 100;MAX_NB_WORDS = 55000;EMBEDDING_DIM = 300;
batch_size_L1 = int(3048/2);batch_size_L2 = int(3048/2);batch_size_L3 = int(3048/2);
L1_model = 2;L2_model = 2;L3_model = 2;rnnType = 4;
trainingBigNetFlag = True;testBigNetFlag = True;
'''

# Config vars
epochs = 15 # Number of epochs to train the main model
level2Epochs = 25 # Number of epochs to train the level 2 models
level3Epochs = 40 # Number of epochs to train the level 3 models
MAX_SEQUENCE_LENGTH = 100 # Maximum sequance lentgh 500 words
MAX_NB_WORDS = 55000 # Maximum number of unique words
EMBEDDING_DIM = 300 # Embedding dimension you can change it to {25, 100, 150, and 300} but need the fasttext version in your directory
TIME_STEPS = MAX_SEQUENCE_LENGTH # Extra variable, TODO: Remove it later
SINGLE_ATTENTION_VECTOR = False # Attention parameter

os.environ['KERAS_BACKEND'] = 'tensorflow'
#MEMORY_MB_MAX = 1200000 # maximum memory you can use
batch_size_L1 = int(3048/2) # batch size in Level 1
batch_size_L2 = int(3048/2) # batch size in Level 2
batch_size_L3 = int(3048/2) # batch size in Level 3

# inputs, Hint: You have to put the following files in the right directory
datasetFileName = "./data/divar_posts_dataset.csv" # original dataset path, train set
phase2DatasetFilename = "./data/phase_2_dataset.csv" # phase 2 dataset path, test set
fastTextDir = './fastText/'

# outputs, Hints: the following files/directories will be created after the first successful code execution
dictionaryFilename = 'wordDict.json' # where to save the extracted words dictionary 
outputPathProcessedDataset = './dataset/' # where to export processed files for later usage
outputPathPhase2DatasetFilename = './dataChallenge/' # where to save processed files for phase2 Dataset
resultFilename = './resultsChallenge2.csv' # where to save results
resultInputFilename = './resultsChallenge2Inputs.csv' # where to save results and inputs
resultsFixLevels = './resultsChallenge2FixLevels.csv' # where to save fixed results, generally better performance
resultsFixLevelsAll = './resultsChallenge2FixLevelsALL.csv' # Check all the samples hierarchy (L1,L2,L3)
resultsTableVis = 'table.html' # where to save all the results alongside inputs for visual judements

#TODO: For now only RNN and CNN is working perfectly, need to change others later
L1_model = 2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
L2_model = 2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2
L3_model = 2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2
rnnType = 4 # rnn model, 0 GRU, 1 Conv + LSTM, 2: RNN+DNN 3: Attention, 4: Big

# Train Controller
#NOTE: for initializing and first run all of the following flags should be true!
preProcessFlag = True # whether to rebuild the dictionary and preprocess datasets

trainingBigNetFlag = True # one Model for all levels (allInONE ;P), Other Flags will be False automatically
testBigNetFlag = True # one Model for all levels, Other Flags will be False automatically
# OR
trainingL1Flag = False # whether to train level 1 model
trainingL2Flag = False # whether to train level 2 models
trainingL3Flag = False # whether to train level 3 models

# Helpers functions
def sanitize(x):
    price = x.split()[-1]
    x = clean_str(x)
    x = text_cleaner(x)
    x = x + price
    x = re.split(r'([a-zA-Z]+)', x)
    x = " ".join(str(item) for item in x)
    return x

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"/", " ", string)
    string = re.sub(r"//", " ", string)
    string = re.sub(r"$NUM", " ", string)
    #string = re.sub(r'[^\w\s]', '', string, re.UNICODE)
    string = re.sub(r'([a-z])\1+', r'\1', string, re.UNICODE)
    return string.strip().lower()

def text_cleaner(text):
    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace(".", " ")
    text = text.replace("&", " ")
    text = text.replace("%", " ")
    text = text.replace("$", " ")
    text = text.replace("#", " ")
    text = text.replace("%", " ")
    text = text.replace("@", " ")
    text = text.replace("!", " ")
    text = text.replace("+", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("،", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.replace("\"", "")
    text = text.replace("-", " ")
    text = text.replace("=", " ")
    text = text.replace("~", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("«", " ")
    text = text.replace("*", " ")
    text = text.replace("❌", " ") 
    text = text.replace("✴", " ")
    text = text.replace("✔", " ")
    text = text.replace("⚽", " ")
    text = text.replace("✅", " ")
    text = text.replace("⌛", " ")
    text = text.replace("⑤", " ")
    text = text.replace("•", " ")
    text = text.replace("♧", " ")
    text = text.replace("num", " ")
    text = text.replace(u'\u2013','')    
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text.lower().replace("num", " ")
    text = re.sub(r'-?\d+\.?\d*', ' ', text)
    text = re.sub(u'\u200c',' ', text)
    text = re.sub(u'\u200e',' ', text)
    text = re.sub(u'\xad',' ', text)
    return text

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def loadData_Tokenizer_Efficient(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM = 100,embedder = 'fastTextEn'):

    pathDataset = './dataset'
    fname = os.path.join(pathDataset,"contextProcessed.txt")
    fname2 = os.path.join(pathDataset,"price.txt")
    fnamek = os.path.join(pathDataset,"Y1.txt")
    fnameL2 = os.path.join(pathDataset,"Y2.txt")
    fnameL3 = os.path.join(pathDataset,"Y3.txt")

    content = pd.read_table(fname, header=None)
    content = content[0].apply(str.strip)
    content2 = pd.read_table(fname2, header=None, dtype='int64') # read price as integer

    contentk = pd.read_table(fnamek, header=None).values
    contentL2 = pd.read_table(fnameL2, header=None).values
    contentL3 = pd.read_table(fnameL3, header=None).values

    Label_L1 = contentk
    Label_L2 = contentL2
    Label_L3 = contentL3

    np.random.seed(7)
    Label = np.column_stack((Label_L1, Label_L2, Label_L3))
    LabelDF = pd.DataFrame(Label)    

    labelsL1 = LabelDF[0].unique()
    labelsL2 = LabelDF[1].unique()
    labelsL3 = LabelDF[2].unique()

    number_of_classes_L1 = len(labelsL1) #number of classes in Level 1
    number_of_classes_L2 = len(labelsL2)
    number_of_classes_L3 = len(labelsL3)    
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(content)
    sequences = tokenizer.texts_to_sequences(content)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    content = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    indices = np.arange(content.shape[0])
    np.random.shuffle(indices)
    content = content[indices]
    Label = Label[indices]
    print(content.shape)

    # join two inputs 
    content = np.concatenate((content,content2),axis=1)

    #TODO: Balance dataset
    X_train, X_test, y_train, y_test  = train_test_split(content, Label, test_size=0.01,random_state= 0, stratify=Label, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.01, random_state=0, stratify=y_train, shuffle=True)
    print(X_train.shape, X_val.shape, X_test.shape)

    embeddings_index = {}
    '''
    For CNN and RNN, we used the text vector-space models using $100$ dimensions as described in Glove. A vector-space model is a mathematical mapping of the word space
    '''
    if embedder == 'glove':
        Glove_path = os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')
        print(Glove_path)
        f = open(Glove_path, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                print("Warnning"+str(values)+" in" + str(line))
            embeddings_index[word] = coefs
        f.close()
        print('Total %s word vectors.' % len(embeddings_index))
    elif embedder == 'fastTextEn':
        embedderName = 'cc.fa.' + str(EMBEDDING_DIM) + '.vec'
        fastText_path = os.path.join(fastTextDir, embedderName)
        print(fastText_path)
        embeddings_index = {}
        with open(fastText_path, encoding='utf8') as infile:
            #for idx,line in enumerate(infile):
            for line in infile:
                #if idx > 1: # skip the first line
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                except:
                    print("Warnning"+str(values)+" in" + str(line))
                if word in wordDict: # need only embedding for words that are in corpus
                    embeddings_index[word] = coefs
        gc.collect()
        print('Total %s word vectors.' % len(embeddings_index))
    return (tokenizer,LabelDF,X_train,y_train,X_val,y_val,X_test,y_test,labelsL1,labelsL2,labelsL3,number_of_classes_L1,number_of_classes_L2,number_of_classes_L3,word_index,embeddings_index)

'''
buildModel_DNN(nFeatures, nClasses, nLayers=3,Numberof_NOde=100, dropout=0.5)
Build Deep neural networks Model for text classification
Shape is input feature space
nClasses is number of classes
nLayers is number of hidden Layer
Number_Node is number of unit in each hidden layer
dropout is dropout value for solving overfitting problem
'''
def buildModel_DNN(Shape, nClasses, nLayers=3,Number_Node=100, dropout=0.5):
    model = Sequential()
    model.add(Dense(Number_Node, input_dim=Shape))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(Number_Node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    return model

'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
word_index in word index , 
embeddings_index is embeddings index, 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lentgh of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding 
'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, type=0, nClasses2 = None, nClasses3= None):
    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    '''
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    if type==0:
        model.add(GRU(100,dropout=0.2, recurrent_dropout=0.2))
    elif type==1:
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(200,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nClasses, activation='softmax'))
    '''
    input1 = Input((MAX_SEQUENCE_LENGTH,),name='context')
    layerM1Embedding = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)(input1)
    input2 = Input((1,),name='price') # price
    if type==0:
        layer = GRU(100,dropout=0.2, recurrent_dropout=0.2)(layerM1Embedding)
    elif type==1:
        layer = GRU(100,dropout=0.2, recurrent_dropout=0.2)(layerM1Embedding)
        layer = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = LSTM(200,dropout=0.2, recurrent_dropout=0.2)(layer)
    elif type==2:
        layerM1 = GRU(100,dropout=0.2, recurrent_dropout=0.2)(layerM1Embedding)
        layerM1 = Dense(nClasses, activation='softmax')(layerM1)
        layerM2 = Dense(nClasses, activation='softmax')(input2)
        layer = Concatenate()([layerM1,layerM2])     
    elif type==3:
        # attention
        attentionMul = attention_3d_block(layerM1Embedding)
        layer = GRU(100,dropout=0.2, recurrent_dropout=0.2)(attentionMul)
    if type==4:
        # attention
        layer = GRU(150,dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(layerM1Embedding)
        #attentionMul = attention_3d_block(layerM1Embedding) # attention before
        #layer = GRU(100,dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(attentionMul)
        #layer = GRU(100,dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(layerM1Embedding)
        #layer = attention_3d_block(layer) # attention after
        #layer = Flatten()(layer) # flatten for dense
        
        out1 = Dense(150,activation='sigmoid')(layer)
        out1 = BatchNormalization()(out1)
        out1 = Dropout(0.5)(out1)
        out1 = Dense(nClasses,activation='softmax',name='out1')(out1)

        out2 = Dense(150,activation='sigmoid')(layer)
        out2 = BatchNormalization()(out2)
        out2 = Dropout(0.5)(out2)
        out2 = Dense(nClasses2,activation='softmax',name='out2')(out2)
        
        out3 = Dense(150,activation='sigmoid')(layer)
        out3 = BatchNormalization()(out3)
        out3 = Dropout(0.5)(out3)
        out3 = Dense(nClasses3,activation='softmax',name='out3')(out3)

        model = Model(inputs=[input1,input2], outputs=[out1,out2,out3],name='BigNet')
        model.summary()
        model.compile(loss={'out1':'sparse_categorical_crossentropy',
                            'out2':'sparse_categorical_crossentropy',
                            'out3':'sparse_categorical_crossentropy'},
                    optimizer='rmsprop',
                    metrics=['acc'])  # rmsprop
    else:
        out = Dense(nClasses, activation='softmax')(layer)
        model = Model(inputs=[input1,input2], outputs=out)
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])  # rmsprop
    return model

'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=0):
word_index in word index , 
embeddings_index is embeddings index,
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum length of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding, 
Complexity we have two different CNN model as follows 
Complexity=0 is simple CNN with 3 hidden layer
Complexity=2 is more complex model of CNN with filter_length of [3, 4, 5, 6, 7]
'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=1):
    if Complexity==0:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedded_sequences = embedding_layer(sequence_input)

        x = Conv1D(256, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(256, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(256, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        preds = Dense(nClasses, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
    else:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        convs = []
        filter_sizes = [3, 4, 5, 6, 7]

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(128, filter_length=fsz, activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs) # Merge(mode='concat', concat_axis=1)(convs)
        l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        l_pool2 = MaxPooling1D(30)(l_cov2)
        l_flat = Flatten()(l_pool2)
        l_dense = Dense(128, activation='relu')(l_flat)
        preds = Dense(nClasses, activation='softmax')(l_dense)
        model = Model(sequence_input, preds)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

    return model

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MAIN CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# https://research.cafebazaar.ir/visage/divar_datasets/
if preProcessFlag:
    print('Load Original Dataset')
    dataset = pd.read_csv(datasetFileName)
    print('Dataset contains: ',dataset.shape)

    # cat1Num 
    dataset['cat1'] = dataset['cat1'].astype('category')
    dataset['cat2'] = dataset['cat2'].astype('category')
    dataset['cat3'] = dataset['cat3'].astype('category')

    newDataset = dataset.copy()

    # save categories for later usage
    cat1Classes = newDataset['cat1'].cat.categories
    cat2Classes = newDataset['cat2'].cat.categories
    cat3Classes = newDataset['cat3'].cat.categories
    print('Saving Dataset Cotegories ...')
    with open('categoriesDivar.json', 'w', encoding='utf8') as outfile:
        json.dump({'cat1':cat1Classes.values.tolist(),'cat2':cat2Classes.values.tolist(),'cat3':cat3Classes.values.tolist()}, outfile, ensure_ascii=False)

    newDataset['Y1'] = newDataset['cat1'].cat.codes
    newDataset['Y2'] = newDataset['cat2'].cat.codes
    newDataset['Y3'] = newDataset['cat3'].cat.codes

    #TODO: max, avg and min sequence length (title and desc)
    #TODO: Build a dictionary, unique words
    newDataset['descLength'] = newDataset['desc'].apply(len)
    newDataset['titleLength'] = newDataset['title'].apply(len)

    print('Desc Length = Mean:',newDataset['descLength'].mean(), 'Min:',newDataset['descLength'].min(), 'Max:',newDataset['descLength'].max())
    print('Title Length = Mean:',newDataset['titleLength'].mean(), 'Min:',newDataset['titleLength'].min(), 'Max:',newDataset['titleLength'].max())

    #newDataset['context'] = newDataset.title + ' ' + newDataset.desc + ' ' + newDataset.desc + ' ' + newDataset.city + ' ' + newDataset.price.astype(str)
    #TODO: Check differenet combinations of hacks
    newDataset['context'] = newDataset.title + ' ' + newDataset.desc + ' ' + newDataset.city + ' ' + newDataset.price.astype(str)
    newDataset.context = newDataset.context.str.replace('\n',' ')

    print('Building a word dictionary ...')
    # build a dictionary
    wordDict = {}
    for idx,row in enumerate(newDataset.context):
        price = row.split()[-1]
        row = clean_str(row)
        row = text_cleaner(row)
        row = row + price
        row = re.split(r'([a-zA-Z]+)', row)
        row = " ".join(str(item) for item in row)
        words = row.split()
        for wrd in words:
            if wrd in wordDict:
                wordDict[wrd] += 1
            else:
                wordDict[wrd] = 1

    print('Word extracted for dictionary: ',len(wordDict))

    # save dictionary to file
    with open(dictionaryFilename, 'w', encoding='utf8') as outfile:
        json.dump(wordDict, outfile, ensure_ascii=False)
        print('Dictionary saved to a file named ', dictionaryFilename)

    newDataset['contextProcessed'] = newDataset.context.apply(lambda row: sanitize(row))

    # export dataset to text files
    print('Exporting processed dataset to external text files ...')
    print('Path: ',outputPathProcessedDataset)
    if not os.path.exists(outputPathProcessedDataset):
        os.makedirs(outputPathProcessedDataset)
    for c in newDataset.columns:
        newDataset[c].to_csv(outputPathProcessedDataset + c + '.txt', index=False)

# # Start: Prepare Dataset for the Model
# load dictionary
print('Loading Dictionary ...')
with open("wordDict.json", "r") as read_file:
    wordDict = json.load(read_file)

print('Loading data ...')
'''
location of input data in two ways 
1: Tokenizer that is using GLOVE or FastText
1: loadData that is using couting words or tf-idf
'''
#X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2,word_index, embeddings_index,number_of_classes_L1 =  \
#        loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
#X_train_DNN, y_train_DNN, X_test_DNN, y_test_DNN, content_L2_Train_DNN, L2_Train_DNN, content_L2_Test_DNN, L2_Test_DNN, number_of_classes_L2_DNN, number_of_classes_L1 = loadData()
tokenizer,LabelDF,X_train,y_train,X_val,y_val,X_test,y_test,labelsL1,labelsL2,labelsL3,number_of_classes_L1,number_of_classes_L2,number_of_classes_L3,word_index,embeddings_index = loadData_Tokenizer_Efficient(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
print("Loading Data is Done")

print('Find all classes of Level2:')
classesL2 = []
for idx in range(0, number_of_classes_L1):
        classes = LabelDF[LabelDF[0]==idx][1].unique()
        classesL2.append(list(classes))
print(classesL2)
# classesL2 = [[11, 3], [18, 13, 1, -1, 10, 22], [12, 25, 6, 26], [4, 0, 20, 23, 5, 24, 15, 17], [2, 9, 8, 16, 14, -1], [7, 21, 19]]

print('Find all classes of Level3:')
classesL3 = []
for idx in range(0, number_of_classes_L2):
        classes = LabelDF[LabelDF[1]==idx][2].unique()
        classesL3.append(list(classes))
print(classesL3)        
# classesL3 = [[21, 11, 6, 22, 0, 47], [61, 62, 51, 8, 39, -1], [42, 54, 13, 12], [-1], [-1], [20], 
# [24, 31, 53, 4, -1], [26, 33], [-1], [14, 48], [38, 41, 17, 32, 44], [40, 28, 3, 7, 49, -1], 
# [50, 1, 60, 5, 10, 34, 55, 52, 57, -1], [-1], [-1], [27, 15, -1], [64, 30, 46, -1], [-1], [36, 37, 56],
# [-1], [25, 43, -1, 58, 19], [-1], [-1], [59, 9, 2, 65], [-1], [16, 23, -1, 18, 35, 63], [-1, 29, 45], []]

#######################RNN Level 1########################
if trainingBigNetFlag:
    classes = pd.unique(LabelDF.values.ravel('K'))
    number_of_classes = len(classes)
    print('Number of Unique Classes',number_of_classes)
    print('Train Big Net, All Levels!')
    print('Create model of RNN')
    print('Sample Label: ',y_train[0,:])
    model = buildModel_RNN(word_index, embeddings_index,len(LabelDF[0].unique()),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,rnnType,len(LabelDF[1].unique()),len(LabelDF[2].unique()))
    classWeights1 = class_weight.compute_class_weight('balanced',np.unique(y_train[:,0]),y_train[:,0])
    classWeights2 = class_weight.compute_class_weight('balanced',np.unique(y_train[:,1]),y_train[:,1])
    classWeights3 = class_weight.compute_class_weight('balanced',np.unique(y_train[:,2]),y_train[:,2])
    model.fit([X_train[:,:-1],X_train[:,-1]], [y_train[:,0],y_train[:,1]+1,y_train[:,2]+1],
        validation_data=([X_val[:,:-1],X_val[:,-1]], [y_val[:,0],y_val[:,1]+1,y_val[:,2]+1]),
        epochs=epochs,
        batch_size=batch_size_L1,
        class_weight=[classWeights1,classWeights2,classWeights3])
    modelFilename = './models/modelBig.h5'
    print('Saving Big Model in',modelFilename)
    model.save(modelFilename)

    trainingL1Flag = False
    trainingL2Flag = False
    trainingL3Flag = False

if trainingL1Flag:
    print('Train Model Level 1')

    if L2_model == 1:
        print('Create model of CNN')
        model = buildModel_CNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
    elif L1_model == 2:
        print('Create model of RNN')
        model = buildModel_RNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,rnnType)
    classWeights = class_weight.compute_class_weight('balanced',np.unique(y_train[:,0]),y_train[:,0])
    model.fit([X_train[:,:-1],X_train[:,-1]], y_train[:,0],
        validation_data=([X_val[:,:-1],X_val[:,-1]], y_val[:,0]),
        epochs=epochs,
        batch_size=batch_size_L1,
        class_weight=classWeights)
    modelL1Filename = './models/modelL1.h5'
    print('Saving Model L1 in',modelL1Filename)
    model.save(modelL1Filename)

if trainingL2Flag:
    print('Training Level 2 Models ...')
    ######################RNN Level 2################################
    for idx in range(0, number_of_classes_L1):
        print('Create Sub model of ', idx)
        classes = LabelDF[LabelDF[0]==idx][1].unique()
        numberSamples = len(LabelDF[LabelDF[0]==idx])
        print(classes, ' Number of samples: ', numberSamples)
        model = Sequential()
        if L2_model == 1:
            print('Create model of CNN')
            model = buildModel_CNN(word_index, embeddings_index,len(classes),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
        if L2_model == 2:
            print('Create model of RNN')
            model = buildModel_RNN(word_index, embeddings_index,len(classes),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,rnnType)
        labelTrain = y_train[y_train[:,0]==idx,1]
        for clsIdx,cls in enumerate(classes):
            labelTrain[labelTrain==cls] = clsIdx
        labelVal = y_val[y_val[:,0]==idx,1]
        for clsIdx,cls in enumerate(classes):
            labelVal[labelVal==cls] = clsIdx
        classWeights = class_weight.compute_class_weight('balanced',np.unique(labelTrain),labelTrain)
        # X_train[y_train[:,0]==idx,:]
        model.fit([X_train[y_train[:,0]==idx,:][:,:-1],X_train[y_train[:,0]==idx,:][:,-1]], labelTrain,
            validation_data=([X_val[y_val[:,0]==idx,:][:,:-1],X_val[y_val[:,0]==idx,:][:,-1]], labelVal),
            epochs=level2Epochs,
            batch_size=batch_size_L2,
            class_weight = classWeights)
        # save model
        modelL2Filename = './models/modelL2_'+ str(idx)+'.h5'
        model.save(modelL2Filename)
        print('Model saved in ',modelL2Filename)
        del model
        K.clear_session()
        gc.collect()

if trainingL3Flag:
    print('Training all models of Level3 ...')
    ######################RNN Level 3################################
    for idx in range(0, number_of_classes_L2):
        print('Create Sub model of ', idx)
        classes = LabelDF[LabelDF[1]==idx][2].unique()
        if len(classes) < 2:
            continue
        numberSamples = len(LabelDF[LabelDF[1]==idx])
        print(classes, ' Number of samples: ', numberSamples)
        model = Sequential()
        if L3_model == 1:
            print('Create model of CNN')
            model = buildModel_CNN(word_index, embeddings_index,len(classes),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
        elif L3_model == 2:
            print('Create model of RNN')
            model = buildModel_RNN(word_index, embeddings_index,len(classes),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,rnnType)
        labelTrain = y_train[y_train[:,1]==idx,2]
        for clsIdx,cls in enumerate(classes):
            labelTrain[labelTrain==cls] = clsIdx
        labelVal = y_val[y_val[:,1]==idx,2]
        for clsIdx,cls in enumerate(classes):
            labelVal[labelVal==cls] = clsIdx
        classWeights = class_weight.compute_class_weight('balanced',np.unique(labelTrain),labelTrain)
        # X_train[y_train[:,1]==idx,:]
        model.fit([X_train[y_train[:,1]==idx,:][:,:-1],X_train[y_train[:,1]==idx,:][:,-1]], labelTrain,
            validation_data=([X_val[y_val[:,1]==idx,:][:,:-1],X_val[y_val[:,1]==idx,:][:,-1]], labelVal),
            epochs=level3Epochs,
            batch_size=batch_size_L3,
            shuffle = True,
            class_weight = classWeights)
        # save model
        modelL3Filename = './models/modelL3_'+ str(idx)+'.h5'
        model.save(modelL3Filename)
        print('Model saved in',modelL3Filename)
        del model
        K.clear_session()
        gc.collect()

print('Find out which class has a trained model: ')
modelExists = glob.glob('./models/*.h5') 
print(modelExists)

print('Start Testing ...')
x = X_test
y = y_test
print('Test has shape of ',x.shape,y.shape)
results = -np.ones_like(y)

if testBigNetFlag:
    # load model
    modelFilename = './models/modelBig'+'.h5'
    model = load_model(modelFilename)
    print('AllInONE Model was loaded ...')
    yPred = model.predict(x[:,:-1], verbose=1, batch_size=2048)
    results[:,0] = yPred[0].argmax(axis=1)
    results[:,1] = yPred[1].argmax(axis=1)-1
    results[:,2] = yPred[2].argmax(axis=1)-1
else:
    # Level 1 Test
    # load model
    modelFilename = './models/modelL1'+'.h5'
    model = load_model(modelFilename)
    # test model
    if rnnType == 2:
        yPred = model.predict([x[:,:-1],x[:,-1]], verbose=1, batch_size=2048)
    else:
        yPred = model.predict(x[:,:-1], verbose=1, batch_size=2048)
    predL1Class = yPred.argmax(axis=1)
    results[:,0] = predL1Class
    print('Test of Level 1 is done')

    # Level 2 Test
    predL1Class = results[:,0]
    for cls in np.unique(predL1Class):
        indexes = predL1Class==cls
        print('Selected Indices for class ',cls,': ',len(indexes[indexes]),'/',len(indexes))
        # load related model 
        modelFilename = './models/modelL2_'+ str(cls)+ '.h5'
        if modelFilename in modelExists:
            model = load_model(modelFilename)  
            if rnnType == 2:
                yPred = model.predict([x[indexes,:][:,:-1],x[indexes,:][:,-1]], verbose=1, batch_size=2048)
            else:
                yPred = model.predict(x[indexes,:][:,:-1], verbose=1, batch_size=2048)
            predClasses = yPred.argmax(axis=1)
            for idx, value in enumerate(classesL2[cls]):
                predClasses[predClasses==idx] = value
            results[indexes,1] = predClasses
            del model
            K.clear_session()
            gc.collect()
        else:
            if len(classesL2[cls]) < 2:
                if classesL2[cls][0]==-1 or classesL2[cls][0] is None:
                    results[indexes,1] = -1
                else:
                    results[indexes,1] = classesL2[cls][0]
    print('Test of Level 2 is done')  

    # Level 3 Test
    predL2Class = results[:,1]
    for cls in np.unique(predL2Class):
        indexes = predL2Class==cls
        print('Selected Indices for class ',cls,': ',len(indexes[indexes]),'/',len(indexes))
        # load related model 
        modelFilename = './models/modelL3_'+ str(cls)+ '.h5'
        if modelFilename in modelExists:
            model = load_model(modelFilename)  
            if rnnType == 2:
                yPred = model.predict([x[indexes,:][:,:-1],x[indexes,:][:,-1]], verbose=1, batch_size=2048)
            else: 
                yPred = model.predict(x[indexes,:][:,:-1], verbose=1, batch_size=2048)
            predClasses = yPred.argmax(axis=1)
            for idx, value in enumerate(classesL3[cls]):
                predClasses[predClasses==idx] = value
            results[indexes,2] = predClasses
            del model
            K.clear_session()
            gc.collect()
        else:
            if cls == -1:
                results[indexes,2] = -1
                continue
            if len(classesL3[cls]) < 2:
                if classesL3[cls][0]==-1 or classesL3[cls][0] is None:
                    results[indexes,2] = -1
                else:
                    results[indexes,2] = classesL3[cls][0]
    print('Test of Level 3 is done')

print('Calculate accuracies: ')
# calculate accuracy
eq = y == results
cat1Acc = len(eq[eq[:,0]==True])/len(eq[:,0])
cat2Acc = len(eq[eq[:,1]==True])/len(eq[:,1])
cat3Acc = len(eq[eq[:,2]==True])/len(eq[:,2])
eqS = eq[eq[:,0]==True,:]
eqS = eqS[eqS[:,1]==True,:]
eqS = eqS[eqS[:,2]==True,:]
totallAcc =  len(eqS)/len(eq)
print('All Categories Acc: ',totallAcc,' Cat1 Acc: ',cat1Acc,' Cat2 Acc: ',cat2Acc,' Cat3 Acc: ',cat3Acc)
# prevRes = [0.8216361556547028 0.9590982191474363 0.8725037883744391 0.831395177051104]

print('Predicting on phase 2 dataset ...')
# Load Test data
dataset = pd.read_csv(phase2DatasetFilename)
print('Phase 2 dataset shape: ',dataset.shape)

newDataset = dataset.copy()
print('Processing phase2 unlabelled data: ')
newDataset['context'] = newDataset.title + ' ' + newDataset.desc + ' ' + newDataset.city + ' ' + newDataset.price.astype(str)
newDataset.context = newDataset.context.str.replace('\n',' ')
#newDataset['context'] = newDataset.title + ' ' + newDataset.desc + ' ' + newDataset.desc + ' ' + newDataset.city + ' ' + newDataset.price.astype(str)
newDataset['contextProcessed'] = newDataset.context.apply(lambda row: sanitize(str(row)))

# export dataset to text files
print('Exporting processed files to ', outputPathPhase2DatasetFilename)
if not os.path.exists(outputPathPhase2DatasetFilename):
    os.makedirs(outputPathPhase2DatasetFilename)
for c in newDataset.columns:
    newDataset[c].to_csv(outputPathPhase2DatasetFilename + c + '.txt', index=False)

print('Load processed context ...')
fname = os.path.join(outputPathPhase2DatasetFilename,"contextProcessed.txt")
fname2 = os.path.join(outputPathPhase2DatasetFilename,"price.txt")
content = pd.read_table(fname, header=None)
content = content[0].apply(str.strip)
content2 = pd.read_table(fname2, header=None, dtype='int64') # read price as integer
word_index = tokenizer.word_index
print('Utilized %s unique tokens.' % len(word_index))
sequences = tokenizer.texts_to_sequences(content)
content = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
content = np.concatenate((content,content2),axis=1)

# all test data
#X_test, y_test
x = content
y = np.zeros((x.shape[0],3), dtype=int)
print('Phase 2 data shape: ', x.shape,y.shape)
results = -np.ones_like(y)

if testBigNetFlag:
    # load model
    modelFilename = './models/modelBig'+'.h5'
    model = load_model(modelFilename)
    print('AllInONE Model was loaded ...')
    yPred = model.predict(x[:,:-1], verbose=1, batch_size=2048)
    results[:,0] = yPred[0].argmax(axis=1)
    results[:,1] = yPred[1].argmax(axis=1)-1
    results[:,2] = yPred[2].argmax(axis=1)-1
else:
    # Level 1 Test
    # load model
    modelFilename = './models/modelL1'+'.h5'
    model = load_model(modelFilename)
    print('Model of Level 1 loaded from ',modelFilename)
    # test model
    if rnnType == 2:
        yPred = model.predict([x[:,:-1],x[:,-1]], verbose=1, batch_size=256)
    else:
        yPred = model.predict(x[:,:-1], verbose=1, batch_size=256)
    predL1Class = yPred.argmax(axis=1)
    results[:,0] = predL1Class
    print('Level 1 is done')

    # Level 2 Test
    predL1Class = results[:,0]
    for cls in np.unique(predL1Class):
        indexes = predL1Class==cls
        print('Selected Indices for class ',cls,': ',len(indexes[indexes]),'/',len(indexes))
        # load related model 
        modelFilename = './models/modelL2_'+ str(cls)+ '.h5'
        if modelFilename in modelExists:
            model = load_model(modelFilename)  
            if rnnType==2:
                yPred = model.predict([x[indexes,:][:,:-1],x[indexes,:][:,-1]], verbose=1, batch_size=256)
            else:
                yPred = model.predict(x[indexes,:][:,:-1], verbose=1, batch_size=256)
            predClasses = yPred.argmax(axis=1)
            for idx, value in enumerate(classesL2[cls]):
                predClasses[predClasses==idx] = value
            results[indexes,1] = predClasses
            del model
            K.clear_session()
            gc.collect()
        else:
            if len(classesL2[cls]) < 2:
                if classesL2[cls][0]==-1 or classesL2[cls][0] is None:
                    results[indexes,1] = -1
                else:
                    results[indexes,1] = classesL2[cls][0]
    print('Level 2 is done')  

    # Level 3 Test
    predL2Class = results[:,1]
    for cls in np.unique(predL2Class):
        indexes = predL2Class==cls
        print('Selected Indices for class ',cls,': ',len(indexes[indexes]),'/',len(indexes))
        # load related model 
        modelFilename = './models/modelL3_'+ str(cls)+ '.h5'
        if modelFilename in modelExists:
            model = load_model(modelFilename)  
            if rnnType==2:
                yPred = model.predict([x[indexes,:][:,:-1],x[indexes,:][:,-1]], verbose=1, batch_size=256)
            else:
                yPred = model.predict(x[indexes,:][:,:-1], verbose=1, batch_size=256)
            predClasses = yPred.argmax(axis=1)
            for idx, value in enumerate(classesL3[cls]):
                predClasses[predClasses==idx] = value
            results[indexes,2] = predClasses
            del model
            K.clear_session()
            gc.collect()
        else:
            if cls == -1:
                results[indexes,2] = -1
                continue
            if len(classesL3[cls]) < 2:
                if classesL3[cls][0]==-1 or classesL3[cls][0] is None:
                    results[indexes,2] = -1
                else:
                    results[indexes,2] = classesL3[cls][0]
    print('Level 3 is done')

print('Preparing the result file ...')
# load categories
with open("categoriesDivar.json", "r") as read_file:
    catDict = json.load(read_file)

# Export to the file
resultsDF = pd.DataFrame(results)
#print(resultsDF.iloc[1:5,:])
resultsDF['cat1'] = resultsDF[0] 
resultsDF['cat2'] = resultsDF[1] 
resultsDF['cat3'] = resultsDF[2] 

for idx,value in enumerate(catDict['cat1']):
    resultsDF.loc[resultsDF[0]==idx, 'cat1'] = catDict['cat1'][idx]
for idx,value in enumerate(catDict['cat2']):
    resultsDF.loc[resultsDF[1]==idx, 'cat2'] = catDict['cat2'][idx]
for idx,value in enumerate(catDict['cat3']):
    resultsDF.loc[resultsDF[2]==idx, 'cat3'] = catDict['cat3'][idx]
resultsDF.loc[resultsDF.cat1==-1,'cat1'] = ''
resultsDF.loc[resultsDF.cat2==-1,'cat2'] = ''
resultsDF.loc[resultsDF.cat3==-1,'cat3'] = ''

resultsDF['title'] = dataset['title']
resultsDF['desc'] = dataset['desc']
resultsDF['contextProcessed'] = newDataset['contextProcessed'] 

# export to file
resultsDF.to_csv(resultFilename,columns=['cat1','cat2','cat3'])
print('Results saved in ', resultFilename)

resultsDF.to_csv(resultInputFilename,columns=['cat1','cat2','cat3','contextProcessed'])
print('Results and Inputs saved in ', resultInputFilename)

# sanitize results
#TODO: A more efficient and faster data structure/method later
if testBigNetFlag:
    print('Santizing the output...')
    oddRes = []
    oddResNum = []
    for idx,sample in enumerate(results):
        if sample[1] in classesL2[sample[0]] and sample[2] in classesL3[sample[1]]:
            continue
        else:
            # something is wrong, prediction hierarchy is not OK!
            oddRes.append(sample)
            oddResNum.append(idx)
    print('Prediction Hierarchy is Problematic in ',len(oddResNum),' Samples!!!')
    # Export to the file
    oddResDF = pd.DataFrame(oddRes)
    oddResDF['idx'] = oddResNum
    oddResDF['cat1'] = oddResDF[0] 
    oddResDF['cat2'] = oddResDF[1] 
    oddResDF['cat3'] = oddResDF[2] 
    for idx,value in enumerate(catDict['cat1']):
        oddResDF.loc[oddResDF[0]==idx, 'cat1'] = catDict['cat1'][idx]
    for idx,value in enumerate(catDict['cat2']):
        oddResDF.loc[oddResDF[1]==idx, 'cat2'] = catDict['cat2'][idx]
    for idx,value in enumerate(catDict['cat3']):
        oddResDF.loc[oddResDF[2]==idx, 'cat3'] = catDict['cat3'][idx]
    oddResDF.loc[oddResDF.cat1==-1,'cat1'] = ''
    oddResDF.loc[oddResDF.cat2==-1,'cat2'] = ''
    oddResDF.loc[oddResDF.cat3==-1,'cat3'] = ''
    # export to file
    oddResFilename = 'oddRes.csv'
    oddResDF.to_csv(oddResFilename,columns=['idx','cat1','cat2','cat3'])
    print('Results saved in ', oddResFilename)

if testBigNetFlag:
    print('Fixing all the problematic samples ...')
    # find unique rows in whole dataset
    # load dataset
    dataset = pd.read_csv(datasetFileName)
    uniqueClasses = dataset[['cat1','cat2','cat3']].drop_duplicates()
    print('Load results')
    resultsOdd = pd.read_csv(oddResFilename)
    results = pd.read_csv(resultInputFilename)
    # Fix the hierarchy based on sublevels, Sublevel prediction are more important to us
    for idx,sample in enumerate(resultsOdd.values):
        cat3X = resultsOdd.iloc[idx]['cat3'] # same as sample
        if type(cat3X) == float and np.isnan(cat3X): # if it is nan
            # go for level 2
            cat2X = resultsOdd.iloc[idx]['cat2']
            if type(cat2X) == float and np.isnan(cat2X):
                # go for level 1
                cat1X = resultsOdd.iloc[idx]['cat1']
                queryText = 'cat1=="{}" & cat2.isnull() & cat3.isnull()'.format(cat1X)
                bestGuess = uniqueClasses.query(queryText)
                if len(bestGuess)==0: continue; # do nothing
                # update the sample in the results
                results.loc[[resultsOdd.iloc[idx].idx],'cat1'] = bestGuess.cat1.values[0]
                results.loc[[resultsOdd.iloc[idx].idx],'cat2'] = bestGuess.cat2.values[0]
                results.loc[[resultsOdd.iloc[idx].idx],'cat3'] = bestGuess.cat3.values[0]
            else:
                queryText = 'cat2=="{}" & cat3.isnull()'.format(cat2X)
                bestGuess = uniqueClasses.query(queryText)
                if len(bestGuess)==0: continue; # do nothing
                # update the sample in the results
                results.loc[[resultsOdd.iloc[idx].idx],'cat1'] = bestGuess.cat1.values[0]
                results.loc[[resultsOdd.iloc[idx].idx],'cat2'] = bestGuess.cat2.values[0]
                results.loc[[resultsOdd.iloc[idx].idx],'cat3'] = bestGuess.cat3.values[0]
        else:
            queryText = 'cat3=="{}"'.format(cat3X)
            bestGuess = uniqueClasses.query(queryText)
            if len(bestGuess)==0: continue; # do nothing
            # update the sample in the results
            results.loc[[resultsOdd.iloc[idx].idx],'cat1'] = bestGuess.cat1.values[0]
            results.loc[[resultsOdd.iloc[idx].idx],'cat2'] = bestGuess.cat2.values[0]
            results.loc[[resultsOdd.iloc[idx].idx],'cat3'] = bestGuess.cat3.values[0]
        print([idx,list(bestGuess['cat1'].values),list(bestGuess['cat2'].values),list(bestGuess['cat3'].values)],end='\r')
    # save fixing dataset
    results.to_csv(resultsFixLevels ,columns=['cat1','cat2','cat3'])
    print('Only problematic samples was solved in ',resultsFixLevels)

    # all 200000 samples
    for idx,sample in enumerate(results.values):
        #print(idx,resultsOdd.iloc[idx])
        cat3X = results.iloc[idx]['cat3'] # same as sample
        if type(cat3X) == float and np.isnan(cat3X): # if it is nan
            # go for level 2
            cat2X = results.iloc[idx]['cat2']
            if type(cat2X) == float and np.isnan(cat2X):
                # go for level 1
                cat1X = results.iloc[idx]['cat1']
                queryText = 'cat1=="{}" & cat2.isnull() & cat3.isnull()'.format(cat1X)
                bestGuess = uniqueClasses.query(queryText)
                if len(bestGuess)==0: continue; # do nothing
                # update the sample in the results
                results.loc[[idx],'cat1'] = bestGuess.cat1.values[0]
                results.loc[[idx],'cat2'] = bestGuess.cat2.values[0]
                results.loc[[idx],'cat3'] = bestGuess.cat3.values[0]
            else:
                queryText = 'cat2=="{}" & cat3.isnull()'.format(cat2X)
                bestGuess = uniqueClasses.query(queryText)
                if len(bestGuess)==0: continue; # do nothing
                # update the sample in the results
                results.loc[[idx],'cat1'] = bestGuess.cat1.values[0]
                results.loc[[idx],'cat2'] = bestGuess.cat2.values[0]
                results.loc[[idx],'cat3'] = bestGuess.cat3.values[0]
        else:
            queryText = 'cat3=="{}"'.format(cat3X)
            bestGuess = uniqueClasses.query(queryText)
            if len(bestGuess)==0: continue; # do nothing
            # update the sample in the results
            results.loc[[idx],'cat1'] = bestGuess.cat1.values[0]
            results.loc[[idx],'cat2'] = bestGuess.cat2.values[0]
            results.loc[[idx],'cat3'] = bestGuess.cat3.values[0]
        print([idx,list(bestGuess['cat1'].values),list(bestGuess['cat2'].values),list(bestGuess['cat3'].values)],end='\r')
    # save fixing dataset
    results.to_csv(resultsFixLevelsAll,columns=['cat1','cat2','cat3'])
    print('All the samples hierarchy are now based on third level only in ',resultsFixLevelsAll)

print('Exporting results for visual judements ...')
#print(resultsDF.iloc[200:300,[3,4,5,6,7]])
#table = tabulate.tabulate(resultsDF.iloc[0:10,[3,4,5,6,7]], headers='keys',tablefmt='psql')
#print(table)
tableHtml = tabulate.tabulate(resultsDF.iloc[:,[3,4,5,8]],headers='keys',tablefmt='html')
f = open(resultsTableVis,'w')
f.write(tableHtml)
f.close()

print('All Categories Acc: ',totallAcc,' Cat1 Acc: ',cat1Acc,' Cat2 Acc: ',cat2Acc,' Cat3 Acc: ',cat3Acc)
