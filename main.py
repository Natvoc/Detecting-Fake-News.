#Dependencias necesarias
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# lectura de datos
df=pd.read_csv('news.csv')

#obtenemos el shape y head
df.shape
df.head()

#obtenemos los labels del DF
labels=df.label
labels.head()

#Dividir el conjunto de datos
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Inicializar TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Ajustar y transformar
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#Inicializar el PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
PassiveAggressiveClassifier(max_iter=50)

#Accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
#print(f'Accuracy: {round(score*100,2)}%')

#Confusion Matrix
resultado = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print (resultado)