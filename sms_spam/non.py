import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
# from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
import scattertext as st
from pprint import pprint


filename = 'sms_data'
sms_df = pd.read_csv(filename, header=None, sep='	', names=['class_sms', 'sms'])

sms_df['sms'] = sms_df['sms'].str.replace(r'\d+', '')
sms_df['sms'] = sms_df['sms'].str.replace(r'\W*\b\w{1,2}\b', '') # regex for replacing words of len less then 2
# sms_df['sms'] = sms_df['sms'].str.replace(r'\d+', '')
sms_df['sms'] = sms_df['sms'].apply(lambda x: x.split())
sms_df['sms'] = sms_df['sms'].apply(lambda x: ','.join(x))
# print(sms_df)
# sms_df.to_csv('new_sms_data3.csv')
# print(len(sms_df.loc[sms_df['class_sms'] == 'spam']))
# sms_df.loc[sms_df['class_sms'] == 'spam'].to_csv('spam_fi2.csv')
# sms_df.loc[sms_df['class_sms']=='spam'].apply(lambda x: x.split())

X = sms_df.sms
y = sms_df.class_sms

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
stopword = stopwords.words('english')
stopword.append('You')
stopword.append('Your')
vect = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b', lowercase=False, ngram_range=(1,2), stop_words=stopword)

X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_class)
false_positive_rate = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[1][1])

print('General Accuracy:', metrics.accuracy_score(y_test, y_pred_class)) # 0.9885
print('False Positive Rate:', false_positive_rate*100)
print (classification_report(y_test, y_pred_class))

spam_token_count = nb.feature_count_[1, :]
X_train_tokens = vect.get_feature_names()
tokens = pd.DataFrame({'token':X_train_tokens, 'spam':spam_token_count})
tokens['spam'] = tokens.spam + 1
tokens['spam'] = tokens.spam / nb.class_count_[1]

# print(tokens.sort_values('spam', ascending=False))  # Common Spam Value Predictor list

false_positive = X_test[(y_test == 'spam') & (y_pred_class == 'ham')]
false_negative = X_test[(y_test == 'ham') & (y_pred_class == 'spam')]
# print(false_positive)
# print(false_negative)

# Calculates the entropy of the given data set for the target attribute.
def entropy(data, class_label, target_attr):
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for i in range(len(data)):
        if class_label[i] == target_attr:
            y = data[i]
            y = y.split(',')
            for word in y:
                if word in val_freq:
                    val_freq[word] += 1
                else:
                    val_freq[word] = 1
                    #     print(val_freq)

                    #     # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


# print(entropy(X, y, 'spam'))

def gain(data, attr, target_attr):
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for i in range(len(data)):
        if class_label[i] == target_attr:
            y = data[i]
            y = y.split(',')
            for word in y:
                if word in val_freq:
                    val_freq[word] += 1
                else:
                    val_freq[word] = 1

                    # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
    entropy_list = []
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        entropy = val_prob * math.log(val_prob, 2)
        entropy_list.append([val, entropy])
    print(entropy_list)