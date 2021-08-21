import pandas as pd
import numpy as np
# import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn import neural_network
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def Normalize(doc):
    # split by white space
    tokens = doc.split()
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens which are non-alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter short tokens
    tokens = [word for word in tokens if len(word) > 1]
    # filter stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return tokens


# Reading the excel file
data = pd.read_csv('blogtext.csv')

data = np.array(data)
n = data[:,0]. size
n = 2000

labels =np.empty(shape=(n) ,dtype='int')

# Data cleaning
#words extracting
vocab = Counter()
for i in range(n):
    labels[i] =data[i, 2]
    tokens = Normalize(data[i, 6].lower())
    vocab.update(tokens)

# generating a histogram of words
word_list = vocab.keys()
vecs = []
for i in range(n):
    print("loading...")
    print(i)
    tokens = Normalize(data[i ,6].lower())
    word_frequency = [tokens.count(w) for w in word_list]
    vecs.append(word_frequency)
vecs = np.array(vecs)


# Preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
vecs = min_max_scaler.fit_transform(vecs)


# making a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.2)

# MLP for regression
model = neural_network.MLPRegressor(hidden_layer_sizes=(50,), random_state=1, max_iter=800)

# Train the model using the training sets
model.fit(X_train, y_train)

# Predict the response for testing the dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier wrong?
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
plt.plot(y_test, linewidth=2, label='Actual age')
plt.plot(y_pred, linewidth=2, label='predicted age')
plt.ylabel('actual and predicted values')
plt.legend(loc='upper right')
plt.show()



