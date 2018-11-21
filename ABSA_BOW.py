import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from nltk import tokenize 
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


# Importing the dataset
dataset = pd.read_table('Data/amazon_reviews_us_Electronics_v1_00.tsv', usecols = ['star_rating','review_body'], header = 0, nrows = 200000, error_bad_lines = False)

lemma = WordNetLemmatizer()
review_sentences = []

for i in range(dataset.shape[0]):
    cleantext = re.sub('<[^<]+?>', '.', str(dataset.review_body[i]).lower())
    cleantext = re.sub('&#[0-9]+;', '', cleantext)
    sentences = tokenize.sent_tokenize(cleantext)
    for sentence in sentences:
        sentence = re.sub('[^\w\s]','',sentence)
        lemma_sent = []
        for word in sentence.split():
            word = lemma.lemmatize(word)
            lemma_sent.append(word)
        review_sentences.append([dataset.star_rating[i],' '.join(lemma_sent)])
        

del dataset

df = pd.DataFrame(review_sentences,columns=['rating','sentence'])

X = df.sentence.tolist()
y = df.rating.values

del df
del review_sentences
# Cleaning the texts

#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
my_stop_words = list(stopwords.words('english'))
#from nltk.stem.porter import PorterStemmer


# Creating the Bag of Words Model
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv = CountVectorizer(max_features = 10000,stop_words = my_stop_words)
#X = cv.fit_transform(review_sentences)

#y = dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y, random_state = 0)


#unique, counts = np.unique(y_train, return_counts=True)
#print(dict(zip(unique, counts)))


# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


vec = TfidfVectorizer(sublinear_tf=True,max_features = 10000, max_df=0.5, ngram_range=(1, 3))
#vec = CountVectorizer(max_features = 10000,stop_words = my_stop_words)
lrg_clf = LogisticRegression()
vec_clf = Pipeline([('vectorizer', vec), ('clf', lrg_clf)])


#classifier = GaussianNB()
#classifier = LogisticRegression()
vec_clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = vec_clf.predict(X_test)




joblib.dump(vec_clf, 'bow_sa_pipeline.joblib') 



#class_probs = classifier.decision_function(X_test)
#class_pred = np.argmax(class_probs,axis=1)

#classifier.classes_
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)

acc_score = accuracy_score(y_test,y_pred)

myReview = ['the bass in these headphones is bad']

print(vec_clf.predict(myReview)[0])




my_x = cv.transform(myReview)

my_y = classifier.predict(my_x)
print(my_y[0])















