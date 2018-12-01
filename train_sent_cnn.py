import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from nltk import tokenize 
import re
import gensim 
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib

ENCODING_DIM = 150
MAX_SENT_WORDS = 50
TOTAL_REVIEWS = 3091103

skip_rows = 1
n_rows = 50000
batch_size = 64
num_classes = 5
epochs = 10
    
for cnt in range(4):
    
    # Importing the dataset
    dataset = pd.read_table('Data/amazon_reviews_us_Electronics_v1_00.tsv',  names = ['star_rating','review_body'],usecols = [7,13], skiprows = skip_rows, nrows = n_rows, error_bad_lines = False)
    
    
    
    #review_sentences = []
    word2vec_sentences = []
    ratings = []
    
    for i in range(dataset.shape[0]):
        cleantext = re.sub('<[^<]+?>', '.', str(dataset.review_body[i]).lower())
        cleantext = re.sub('&#[0-9]+;', '', cleantext)
        sentences = tokenize.sent_tokenize(cleantext)
        for sentence in sentences:
            sentence = re.sub('[^\w\s]','',sentence)
            word2vec_sentences.append(sentence.split())
            #sentence_lengths.append(len(sentence.split()))
            ratings.append(dataset.star_rating[i])
            #review_sentences.append([dataset.star_rating[i],sentence])
    
    
    
    del dataset
    
    #df = pd.DataFrame(review_sentences,columns=['rating','sentence'])
    
    #review_sentences = df.sentence.tolist()
    
    #unique, counts = np.unique(sentence_lengths, return_counts=True)
    #print(dict(zip(unique, counts)))
    
    
    # build vocabulary and train model
    
    if cnt==0:
        w2v_model = gensim.models.Word2Vec(
            word2vec_sentences,
            size=ENCODING_DIM,
            window=5,
            min_count=1,
            workers=5)
        w2v_model.train(word2vec_sentences, total_examples=len(word2vec_sentences), epochs=10)
    else:    
        w2v_model.build_vocab(word2vec_sentences, update=True)
        w2v_model.train(word2vec_sentences, total_examples=len(word2vec_sentences), epochs=10)

    
    
    
    
    #print(w2v_model.wv.vocab.keys())
    #model.wv.most_similar (positive=['computer'])
    #model.wv['described']
    
    X = np.zeros((len(word2vec_sentences), MAX_SENT_WORDS,ENCODING_DIM))
    for sdx, sentence in enumerate(word2vec_sentences):
        for wdx, word in enumerate(sentence):
            if wdx >= MAX_SENT_WORDS:
                break
            X[sdx,wdx,:] = w2v_model.wv[word]
                
        
        
    y = np.asarray(ratings)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    
    
    #del word2vec_sentences
    #del w2v_model
    del ratings
    
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y, random_state = 0)
    
    
    #unique, counts = np.unique(y_train, return_counts=True)
    #print(dict(zip(unique, counts)))
    
    
    # Feature Scaling
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)
    
    
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        print('channels_first')
        input_shape = (1, X.shape[1], X.shape[2])
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else:
        print('channels_last')
        input_shape = (X.shape[1], X.shape[2],1)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:int(X.shape[0]*.8)], indices[int(X.shape[0]*.8):]
    X_train, X_test, y_train, y_test = X[training_idx,:,:,:], X[test_idx,:,:,:], y[training_idx,:], y[test_idx,:]
    
    del X
    
    #input_shape = (MAX_SENT_WORDS, ENCODING_DIM, 1)
    
    # convert class vectors to binary class matrices
    
    if cnt == 0:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
    
    
    model.fit(X_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test)
              )
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    skip_rows = skip_rows + 50000
    
model.save('cnn_absa_model.h5') 
joblib.dump(lb, 'cnn_label_binarizer.joblib') 
w2v_model.save("word2vec.model")

    
