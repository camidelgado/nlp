from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
#convertir texto en un valor a una matrix, cada columna es una palabra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import unicodedata
import pickle
import re
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')

df = df.drop_duplicates().reset_index(drop = True)
df['is_spam'] = df['is_spam'].astype(int)
df['url']=df['url'].str.strip()
df['url']=df['url'].str.lower()

df['url']=df['url'].str.replace('!','')
df['url']=df['url'].str.replace(',','')
df['url']=df['url'].str.normalize('NFKC') #ANALIZA TEXTO NO LATINO Y LO TRATA DE ARREGLAR
df['url']=df['url'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True)

def normalize_string(text_string):
    if text_string is not None:
        result=unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else: 
        result=None
    return result

df['url']=df['url'].apply(normalize_string)

def url(text):
    return re.sub(r'(https://www|https://)', '', text)

def comas(text):
    """
    Elimina comas del texto
    """
    return re.sub(',', ' ', text)

def espacios(text):
    """
    Elimina enters dobles por un solo enter
    """
    return re.sub(r'(\n{2,})','\n', text)

def minuscula(text):
    """
    Cambia mayusculas a minusculas
    """
    return text.lower()

def numeros(text):
    """
    Sustituye los numeros
    """
    return re.sub('([\d]+)', ' ', text)

def caracteres_no_alfanumericos(text):
    """
    Sustituye caracteres raros, no digitos y letras
    Ej. hola 'pepito' como le va? -> hola pepito como le va
    """
    return re.sub("(\\W)+"," ",text)

def comillas(text):
    """
    Sustituye comillas por un espacio
    Ej. hola 'pepito' como le va? -> hola pepito como le va?
    """
    return re.sub("'"," ", text)

def palabras_repetidas(text):
    """
    Sustituye palabras repetidas

    Ej. hola hola, como les va? a a ustedes -> hola, como les va? a ustedes
    """
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def esp_multiple(text):
    """
    Sustituye los espacios dobles entre palabras
    """
    return re.sub(' +', ' ',text)

df['url_limpia'] = df['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)

vec = CountVectorizer().fit_transform(df['url_limpia'])

X_train, X_test, y_train, y_test = train_test_split(vec, df['is_spam'], stratify = df['is_spam'], random_state = 207)

classifier = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(random_state=1234),param_grid,verbose=2)
grid.fit(X_train,y_train)
predictions = grid.best_estimator_.predict(X_test)