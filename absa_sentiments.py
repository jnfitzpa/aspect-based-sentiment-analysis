from sklearn.externals import joblib
from keras import backend as K
from keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import re

ENCODING_DIM = 150
MAX_SENT_WORDS = 50
num_classes = 5

#sentences = ['this does not work', 'this is an ok product']

def sentiments(sentences):
    model = load_model('cnn_absa_model.h5')
    lb = joblib.load('cnn_label_binarizer.joblib') 
    w2v_model = Word2Vec.load('word2vec.model')
    vec_clf = joblib.load('bow_sa_pipeline.joblib')
    
    bow_pred = vec_clf.predict(sentences)
    
    word2vec_sentences = []
    
    for sentence in sentences:
        sentence = re.sub('[^\w\s]','',sentence.lower())
        word2vec_sentences.append(sentence.split())
        
                
    X = np.zeros((len(word2vec_sentences), MAX_SENT_WORDS,ENCODING_DIM))
    for sdx, sentence in enumerate(word2vec_sentences):
        for wdx, word in enumerate(sentence):
            if wdx >= MAX_SENT_WORDS:
                break
            X[sdx,wdx,:] = w2v_model.wv[word]           
    
    
    if K.image_data_format() == 'channels_first':
        #input_shape = (1, X.shape[1], X.shape[2])
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else:
        #input_shape = (X.shape[1], X.shape[2],1)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    y_pred = model.predict_classes(X)
    y_pred_bin = np.zeros((y_pred.shape[0],num_classes))
    
    for i in range(y_pred.shape[0]):
        y_pred_bin[i,y_pred[i]] = 1
        
    y_pred_new = lb.inverse_transform(y_pred_bin)   
        
    #print(y_pred_new)
    
    return np.rint(np.average(np.asarray([y_pred_new,bow_pred]),axis=0))
