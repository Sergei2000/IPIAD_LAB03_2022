from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



# категории для классификации

categories = ['comp.graphics', 'rec.autos','sci.crypt','soc.religion.christian','talk.politics.mideast']

twenty_train = fetch_20newsgroups(subset='train',categories = categories, shuffle=True, random_state=42)
count_vect = CountVectorizer()
X_vectors = count_vect.fit_transform(twenty_train.data)

print("50/50 devision:")
X_train,X_test,Y_train, Y_test = train_test_split(X_vectors.toarray(),twenty_train.target,test_size=0.5)
print("studying Gaussian model")
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
print("now it is clever")
print("Gaus Model accurancy is : ",gnb.score(X_test,Y_test))



svc = SVC(C=1.0, kernel='linear', degree=3,gamma="auto")
print("Studying model")
svc.fit(X_train,Y_train)
print("model is clever")
prediction = svc.predict(X_test)
print("Svc Model accurancy is : ",svc.score(X_test,Y_test))

print("75/25 devision:")

X_train,X_test,Y_train, Y_test = train_test_split(X_vectors.toarray(),twenty_train.target,test_size=0.25)
print("studying Gaussian model")
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
print("now it is clever")
print("Gaus Model accurancy is : ",gnb.score(X_test,Y_test))


svc = SVC(C=1.0, kernel='linear', degree=3,gamma="auto")
print("Studying model")
svc.fit(X_train,Y_train)
print("model is clever")
prediction = svc.predict(X_test)
print("Svc Model accurancy is : ",svc.score(X_test,Y_test))



