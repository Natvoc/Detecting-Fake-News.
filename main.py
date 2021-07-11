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
