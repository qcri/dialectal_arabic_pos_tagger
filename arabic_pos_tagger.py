#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this script uses pretrained model to segment Arabic dialect data.
# it takes the pretrained model trained on joint dialects and the 
# training vocab and produces segmented text
#
# Copyright (C) 2017, Qatar Computing Research Institute, HBKU, Qatar
# Las Update: Mon Jul  2 13:47:27 +03 2018
# Ahmed Abdelali aabdelali at qf dot org dot qa
#
# @InProceedings{DARWISH18.562,
#  author = {Kareem Darwish and Hamdy Mubarak and Ahmed Abdelali and Mohamed Eldesouki and Younes Samih and Randah Alharbi and Mohammed Attia and Walid Magdy and Laura Kallmeyer},
#  title = "{Multi-Dialect Arabic POS Tagging: A CRF Approach}",
#  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
#  year = {2018},
#  month = {May 7-12, 2018},
#  address = {Miyazaki, Japan},
#  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
#  publisher = {European Language Resources Association (ELRA)},
#  isbn = {979-10-95546-00-9},
#  language = {english}
# }
#

"""
@author: Abdelali
@author: Samih
"""
import os
import pickle
import numpy as np
import argparse
import sys
import unicodedata
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.layers.core import Dense, Dropout, Lambda, Masking
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adamax,Adam,SGD,RMSprop
from keras.regularizers import l2
from keras.layers import ChainCRF


seed = 12345
np.random.seed(seed)

def save_pickled_file(obj, name):
    f = open(name + '.pkl', 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    
def load_pickled_file(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def readFile(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(path,"r"):
        line = line.rstrip()
        if(line.startswith('#') or line.startswith('@') or line.startswith('http:')):
            continue
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
            sentence = []

    return sentences

def readTestFile(path):
    """
    Load sentences.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    nline = 0
    nsegm = 0
    nword = 0
    for line in open(path,"r"):

        line = line.rstrip()
        words = line.split(' ')
        nline += 1
        for word in words:
            if(word.startswith('#') or word.startswith('@') or word.startswith('http:')):
                continue
            nword += 1
            if not word:
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0]:
                        sentences.append(sentence)
                    sentence = []
            else:
                word = word.split('+')
                nw = 0
                for w in word:
                    sentence.append(w)
                    nsegm += 1
                    nw += 1
                    if(nw < len(word)):
                        sentence.append('TB')
                sentence.append('WB')
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0]:
                sentences.append(sentence)
                sentence = []

    print("Stats: Number of Lines %d, Number of Words: %d, Number of Segments: %d"%(nline,nword,nsegm))

    return sentences

def getLabels(filepath):
    labels = []
    for line in open(filepath, "r"):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    return list(set(labels))

def getVocabs(filepath):
    vocabs = []
    for line in open(filepath, "r"):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        vocabs.append(splits[0].strip())
    vocabs.append('UNKNOWN')
    vocabs.append('PADDING')

    return list(set(vocabs))


def normalizeWord(line):
    #line = unicode(line)  # Convert to UTF8
    line = unicodedata.normalize('NFKD', line).encode('ASCII', 'ignore')
    return line.strip()

"""
Functions to read in the files from the pos corpus,
create suitable numpy matrices for train/dev/test
"""
def createNumpyArray(sentences, windowsize, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']
    xMatrix = []
    yVector = []
    wordCount = 0
    unknownWordCount = 0
    for sentence in sentences:
        targetWordIdx = 0
        for targetWordIdx in range(len(sentence)):
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            for wordPosition in range(targetWordIdx - windowsize, targetWordIdx + windowsize + 1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    continue
                word = sentence[wordPosition][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                wordIndices.append(wordIdx)
            # Get the label and map to int
            labelIdx = label2Idx.get(sentence[targetWordIdx][1].strip())
            #print (labelIdx , wordIndices)
            xMatrix.append(wordIndices)
            yVector.append(labelIdx)
    print("Unknowns: %.2f" % (unknownWordCount / (float(wordCount)) * 100))
    return np.asarray(xMatrix, dtype='int32'), np_utils.to_categorical(np.array(yVector, dtype='int32'))

def createTestArray(sentences, windowsize, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN']
    paddingIdx = word2Idx['PADDING']
    xMatrix = []
    yVector = []
    wordCount = 0
    unknownWordCount = 0
    for sentence in sentences:
        targetWordIdx = 0
        for targetWordIdx in range(len(sentence)):
            # Get the context of the target word and map these words to the index in the embeddings matrix
            wordIndices = []
            for wordPosition in range(targetWordIdx - windowsize, targetWordIdx + windowsize + 1):
                if wordPosition < 0 or wordPosition >= len(sentence):
                    wordIndices.append(paddingIdx)
                    continue
                word = sentence[wordPosition]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif normalizeWord(word) in word2Idx:
                    wordIdx = word2Idx[normalizeWord(word)]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                wordIndices.append(wordIdx)

            #print labelIdx , wordIndices
            xMatrix.append(wordIndices)
    print("Unknowns: %.2f" % (unknownWordCount / (float(wordCount)) * 100))
    return np.asarray(xMatrix, dtype='int32')


def computeWordLevelAccuracy(y_true, y_pred, mapping):
    y_true_word = []
    y_pred_word = []
    tokenTrue = ''
    tokenPred = ''
    correct = 0
    total = 0
    for i in range(len(y_true)):
        #print(i,y_true[i], y_pred[i],mapping[y_pred[i]])
        if y_true[i] == 'TB' or y_true[i] == 'WB':
            y_true_word.append(tokenTrue)
            y_pred_word.append(tokenPred)
            if tokenTrue == tokenPred:
                correct += 1
            total += 1
            tokenTrue = ''
            tokenPred = ''
        elif y_true[i] == 'EOS':
            continue
        else:
            if len(tokenTrue) > 0:
                tokenTrue += '+'
                tokenPred += '+'
            tokenTrue += y_true[i]
            tokenPred += mapping[y_pred[i]]
    #print(correct, total, 1.0*correct/total)
    return 1.0*correct/total, np.array(y_true_word), np.array(y_pred_word)

def getWords(x_src,y_pred, mapping):
    y_pred_word = []
    x_pred_word = []
    tokenPred = ''
    tokenTrue = ''
    mapping[len(mapping)]='UNK'
    total = 0
    for i in range(len(y_pred)):
        #print(i,x_src[i],y_pred[i],tokenPred, len(mapping))
        #print(mapping[y_pred[i]])
        if x_src[i] == 'TB' or mapping[y_pred[i]] == 'TB' or mapping[y_pred[i]] == 'WB':
            y_pred_word.append(tokenPred)
            x_pred_word.append(tokenTrue)
            total += 1
            tokenPred = ''
            tokenTrue = ''
        elif mapping[y_pred[i]] == 'PUNC' or mapping[y_pred[i]] == 'EMOT':
            if len(tokenPred) > 0:
                y_pred_word.append(tokenPred)
                x_pred_word.append(tokenTrue)
                total += 1
            y_pred_word.append(x_src[i])
            x_pred_word.append(mapping[y_pred[i]])
            total += 1
            tokenPred = ''
            tokenTrue = ''            
        elif mapping[y_pred[i]] == 'EOS':
            continue
        else:
            if len(tokenPred) > 0:
                tokenPred += '+'
                tokenTrue += '+'
            tokenPred += mapping[y_pred[i]]
            print("==>",tokenTrue,"::",x_src[i])
            tokenTrue += x_src[i]
    return x_pred_word, y_pred_word

def main():

    # parse user input
    parser = argparse.ArgumentParser()

    #file related args
    parser.add_argument("-m", "--model-dir",   default="./models/", help="directory to save the best models")

    parser.add_argument("-t", "--train-set",   default="./data/EG.txt-train.txt", help="maximul sentence length (for fixed size input)") # 
    parser.add_argument("-v", "--dev-set",     default="./data/EG.txt-dev.txt", help="source vocabulary size") # 
    parser.add_argument("-s", "--test-set",    default="./data/EG.txt-test.txt", help="target vocabulary size") # 

    parser.add_argument("-i", "--input",    default="./data/EG.txt-test.sample-eng.txt", help="a sample input segmened file") # 
    parser.add_argument("-o", "--output",    default="", help="POS output") # 

    # network related
        #input
    parser.add_argument("-e", "--emb-size",    default=300, type=int, help="dimension of embedding") # emb matrix col size
    parser.add_argument("-w", "--window-size", default=10, type=int, help="dimension of embedding") #
    parser.add_argument("-d", "--vocab-emb",   default="./data/segmented-vectors", help="vocabulatry pre-trained embeddings") #
    parser.add_argument("-r", "--final_layer", default="lstm", help="Final optimization layer 'crf' or 'lstm'") 

    #learning related
    parser.add_argument("-a", "--learning-algorithm",  default="adam", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta)")
    parser.add_argument("-b", "--batch-size",  default=128, type=int, help="batch size")
    parser.add_argument("-n", "--epochs",      default=100, type=int, help="nb of epochs")

    #others
    parser.add_argument("-V", "--verbose-level",default=1, type=int, help="verbosity level (0 < 1 < 2)")
    parser.add_argument("-g", "--showGraph",    default=False,   help="show precision and accuracy graphs") # 
    parser.add_argument("-l", "--train-model",  default=False, type=lambda x: (str(x).lower() == 'true'),  help="Train the model, default False")

    
    parser.parse_args()

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # 5 to the left, 5 to the right
    windowSize = args.window_size

    print("Pos with Keras, only token, window size %d" % (windowSize))
    print("Train the model: %s"%(args.train_model))

    # Read in the vocab
    #print("Read in the vocab")
    vocabPath = args.vocab_emb

    word2Idx = {}  # Maps a word to the index in the embeddings matrix
    idx2word = {}
    embeddings = []  # Embeddings matrix

    with open(vocabPath, 'r') as fIn:
        idx = 0
        for line in fIn:
            split = line.strip().split(' ')
            embeddings.append(np.array([float(num) for num in split[1:]]))
            word2Idx[split[0]] = idx
            idx += 1

    idx2word = {v: k for k, v in word2Idx.items()}

    embeddings = np.asarray(embeddings, dtype='float32')

    embedding_size = embeddings.shape[1]

    # Create a mapping for our labels
    labels_list = getLabels(args.train_set)
    labels_list = set(labels_list + getLabels(args.dev_set))

    label2Idx = dict((l, i) for i, l in enumerate(labels_list))
    idx2Label = {v: k for k, v in label2Idx.items()}


    if(args.train_model == False):
        word2Idx = load_pickled_file(args.model_dir+'/word2Idx')
        label2Idx = load_pickled_file(args.model_dir+'/label2Idx')
        idx2Label = {v: k for k, v in label2Idx.items()}
    elif(not os.path.isfile(args.model_dir+'/list2idx.pkl')):
        save_pickled_file(word2Idx, args.model_dir+'/word2Idx')
        save_pickled_file(label2Idx, args.model_dir+'/label2Idx')

    print("Idx2Label:",idx2Label)
        
    if(args.train_model == True):
        # Read in data
        print("Read in data and create matrices")
        train_sentences = readFile(args.train_set)
        dev_sentences = readFile(args.dev_set)
        test_sentences = readFile(args.test_set)
    else:
        test_sentences = readTestFile(args.input)

    test_src = []
    test_trg = []
    for sentence in test_sentences:
        for word in sentence:
            if(args.train_model == True):
                test_src.append(word[0])
                test_trg.append(word[1])
            else:
                test_src.append(word.split('\t')[0])

    if(args.train_model == True):
        # Create numpy arrays
        X_train, y_train = createNumpyArray(train_sentences, windowSize, word2Idx, label2Idx)
        X_dev, y_dev = createNumpyArray(dev_sentences, windowSize, word2Idx, label2Idx)
        X_test, y_test = createNumpyArray(test_sentences, windowSize, word2Idx, label2Idx)
    else:
        X_test = createTestArray(test_sentences, windowSize, word2Idx, label2Idx)

	#print(test_src)
    
    # Create the  Network

    n_in = 2 * windowSize + 1
    n_out = len(label2Idx)
    batch_size = args.batch_size
    epochs = args.epochs


    # If CRF change Tensor to shape '(?, ?, ?)'
    if(args.final_layer == 'crf'):
        maxlen = n_in
        if(args.train_model == True):
            X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
            y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
            y_train = np.expand_dims(y_train, -1)

            X_dev = sequence.pad_sequences(X_dev, maxlen=maxlen, padding='post')
            y_dev = sequence.pad_sequences(y_dev, maxlen=maxlen, padding='post')
            y_dev = np.expand_dims(y_dev, -1)

            X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')
            y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post')
            y_test = np.expand_dims(y_test, -1)
        else:
            X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')


    print ('number of classes:', n_out)
    print ("Embeddings shape", embeddings.shape)
    print ("input dim", embeddings.shape[0],embeddings.shape[1])



    if(args.final_layer == 'crf'):
        model = Sequential()
        model.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in, weights=[embeddings],trainable=False))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(300,return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(n_out)))
        crf = ChainCRF()
        model.add(crf)
        model.compile(loss=crf.sparse_loss, optimizer= RMSprop(0.01), metrics=['sparse_categorical_accuracy'])
    else:
        model = Sequential()
        model.add(Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0], input_length=n_in, weights=[embeddings],trainable=False))

        model.add(Dropout(0.5))
        #model.add(LSTM(300, return_sequences=False))
        model.add(Bidirectional(LSTM(embedding_size, return_sequences=False)))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim=n_out, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=args.learning_algorithm, metrics=['accuracy'])       

    model.summary()

    if(os.path.isfile(args.model_dir+'/keras_weights.hdf5')):
    	model.load_weights(args.model_dir+'/keras_weights.hdf5')

    if(args.train_model == True):
        early_stopping = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(args.model_dir+"/keras_weights.hdf5",verbose=1,save_best_only=True)

        history = model.fit(X_train, y_train,
               batch_size=batch_size,
               #epochs=epochs,
               nb_epoch=epochs,
               verbose=1,
               shuffle=True,
               callbacks=[early_stopping, checkpointer],
               validation_data=[X_dev, y_dev])

    model.load_weights(args.model_dir+'/keras_weights.hdf5')

    if(args.train_model == True):
        preds_dev = model.predict_classes(X_dev, batch_size=64, verbose=0)
        if(args.final_layer == 'crf'):
            preds_dev = preds_dev.argmax(-1)

    if(args.final_layer == 'crf'):
        preds_test = model.predict_classes(X_test, batch_size=512, verbose=0).argmax(-1)
    else:
        preds_test = model.predict_classes(X_test, batch_size=512, verbose=0)


    # print("test_src:",len(test_src))
    # print("X_test", len(X_test))
    # print("preds_test",len(preds_test))
    if(args.output !=''):
    	fout = open(args.output , 'w')
    else:
    	fout = sys.stdout

    for w,p in zip(test_src,preds_test):
        #print("W:",w," P:",p)
        fout.write(w+'\t'+(idx2Label[p] if(p<len(idx2Label)) else 'UNKNOWN')+'\n') 


    #print(score_test[1])
    if(args.train_model == True):
        from sklearn.metrics import confusion_matrix, classification_report
        score_test = model.evaluate(X_test, y_test, batch_size=500)
        print ("Test Score:",score_test[1])
        score_dev = model.evaluate(X_dev, y_dev, batch_size=500)
        print ("Dev Score:",score_dev[1])

        print('')
        print(classification_report(np.argmax(y_dev, axis=1), preds_dev, target_names=labels_list))

        if(args.showGraph):
            print('')
            print(confusion_matrix(np.argmax(y_dev, axis=1), preds_dev))

            print('')
            print(classification_report(np.argmax(y_test, axis=1), preds_test, target_names=labels_list))
            print('')
            print(confusion_matrix(np.argmax(y_test, axis=1), preds_test))

            # # list all data in history
            print(history.history.keys())
            import matplotlib.pyplot as plt
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            #summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.show()

        score, y_true_word, y_pred_word = computeWordLevelAccuracy(test_trg, preds_test, idx2Label)
        print(score)

if __name__ == "__main__":
    main()


