import csv
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.utils import resample



class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

class Experiment:
    def __init__(self, input_file, neg_handling, fix_typo_alay, classifier, tfidf, positive_words, negative_words, harsh, positive_prop, negative_prop, harsh_prop, char_ngram, word_ngram, k_cross_val):
        self.input_file = input_file
        self.classifier_name = classifier
        self.tfidf = tfidf
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.harsh = harsh
        self.positive_prop = positive_prop
        self.negative_prop = negative_prop
        self.harsh_prop = harsh_prop
        self.char_ngram = char_ngram
        self.word_ngram = word_ngram
        self.k_cross_val = k_cross_val
        self.positive_sentiment = []
        self.negative_sentiment = []
        self.harsh_words = []
        self.typo_dict = {}
        self.alay_dict = {}
        self.neg_handling = neg_handling
        self.fix_typo_alay = fix_typo_alay
        self.negation_word = ['tidak', 'tdk', 'tak', 'ga', 'gak', 'gk', 'enggak', 'engga', 'bukan', 'bkn',
                            'jangan', 'jgn', 'belum', 'blm', 'blum', 'blom']

    def load_data(self):
        self.df = pd.read_csv(self.input_file)
        self.df.dropna(axis=0)
    
    def load_stopwords(self, language='indonesian'):
        self.stop_words = set(stopwords.words(language))

    def load_negative_words(self, filename='data/negatif-indonesia.txt'):
        with open(filename) as f:
            data = f.readlines()

        for w in data:
            w = w.replace('\n', '')
            self.negative_sentiment.append(w)

    def load_positive_words(self, filename='data/positif-indonesia.txt'):
        with open(filename) as f:
            data = f.readlines()

        for w in data:
            w = w.replace('\n', '')
            self.positive_sentiment.append(w)

    def load_harsh_words(self, filename='data/kata-kasar.txt'):
        with open(filename) as f:
            data = f.readlines()

        for w in data:
            w = w.replace('\n', '')
            self.harsh_words.append(w)

    def count_proportion(self, sentiment_count, words):
         return float(sentiment_count / words)

    def extract_words(self, data):
        tokens = data.split(' ')
        for t in tokens:
            t = re.sub(r'[^\w\s]','', t)
            t = re.sub(r'[^a-zA-Z]','', t)
            t.strip()

        return tokens

    def load_typo_dict(self):
        with open('data/ID-Kamus-Typo-txt.txt') as f:
            self.typo_dict = dict(x.rstrip().split(None, 1) for x in f)

    def fix_typo(self, data):
        # split kalimat jadi per kata by space. bisa jg klo mau pke tokenize. sila diubah2
        words = data.split(" ")
        
        # iterasi setiap katanya, klo ada di dict kata typo, di replace sm yg bener
        for word in words:
            if word in self.typo_dict:
                if self.typo_dict[word][0] == '_':
                    continue
                data = data.replace(word, self.typo_dict[word])
        return data

    def load_kamus_alay(self):
        with open('data/kamusalay.csv', mode='r') as infile:
            reader = csv.reader(infile, delimiter=",")
            self.alay_dict = {rows[0]:rows[1] for rows in reader}

    def negation_handling(self, data):
        neg_word_found = False
        data = data.split(' ')
        words = []
        for w in data:
            # print("w", w)
            if neg_word_found:
                w = 'NEG_' + w
                neg_word_found = False
            if w in self.negation_word:
                neg_word_found = True
                continue
            words.append(w)
        return ' '.join(words)

    def processing(self):
        #lowering
        self.df['processed'] = self.df['text'].apply(lambda x: x.lower())
        self.df['processed'] = self.df['processed'].apply(lambda x: x.replace('\n', ' '))

        self.df['processed'] = self.df['processed'].apply(lambda x: x.replace('#', ' #'))
        self.df['tokens'] = self.df['processed'].apply(lambda x: self.extract_words(x))

        self.df['processed'] = self.df['tokens'].apply(lambda tokens: ' '.join([t.strip() for t in tokens if len(t.strip()) >  0 
                                                                                                            and (t.strip()[:4] != 'http' 
                                                                                                                or t.strip()[1] != '#'
                                                                                                                ) 
                                                                                                            # and t.strip() not in self.stop_words
                                                                                                            ]))#

        # #removing punctuation
        self.df['processed'] = self.df['processed'].apply(lambda x: re.sub(r'[^\w\s]','', x))
        if self.fix_typo_alay:
            # fix typo
            self.df['processed']= self.df['processed'].apply(lambda x: self.fix_typo(x))
            # # fix alay words
            my_filter = lambda t: ''.join(self.alay_dict.get(word, word) for word in re.split( '(\W+)', t))
            self.df['processed'] = self.df['processed'].apply (my_filter)
        
        if self.neg_handling:
            self.df['processed'] = self.df['processed'].apply(lambda x: self.negation_handling(x))
        self.df['words'] = self.df['processed'].apply(lambda x: len(self.df['tokens']))
        
        self.df['positive_count'] = self.df['processed'].apply(lambda x: len([t for t in x.split(' ') if t in self.positive_sentiment]))
        self.df['negative_count'] = self.df['processed'].apply(lambda x: len([t for t in x.split(' ') if t in self.negative_sentiment]))
        self.df['harsh_count'] = self.df['processed'].apply(lambda x: len([t for t in x.split(' ') if t in self.harsh_words]))

        self.df['positive_proportion'] = self.df.apply(lambda x: self.count_proportion(x['positive_count'], x['words']), axis=1)
        self.df['negative_proportion'] = self.df.apply(lambda x: self.count_proportion(x['negative_count'], x['words']), axis=1)
        self.df['harsh_proportion'] = self.df.apply(lambda x: self.count_proportion(x['harsh_count'], x['words']), axis=1)

    def data_balancing(self):
        self.df_majority = self.df[self.df.label==0]
        self.df_minority = self.df[self.df.label==1]

        # Upsample majority class
        self.df_minority_upsampled = resample(self.df_minority, 
                                         replace=True,    # sample without replacement
                                         n_samples=674,     # to match minority class
                                         random_state=123) # reproducible results236
         
        self.df = pd.concat([self.df_minority_upsampled, self.df_majority])


    def extract_features(self):
        feature_list = []

        if self.tfidf:
            text = Pipeline([
                    ('selector', TextSelector(key='processed')),
                    ('tfidf', TfidfVectorizer(stop_words=list(self.stop_words)))
                ])
            feature_list.append(('text', text))

        if self.positive_words:
            print("Extracting positive words")
            positive_count =  Pipeline([
                ('selector', NumberSelector(key='positive_count')),
                ('standard', StandardScaler()),
            ])
            feature_list.append(('positive_words', positive_count))

        if self.negative_words:
            print("Extracting negative words")
            negative_count =  Pipeline([
                ('selector', NumberSelector(key='negative_count')),
                ('standard', StandardScaler()),
            ])
            feature_list.append(('negative_words', negative_count))

        if self.harsh:
            print("Extracting harsh words")
            harsh_count =  Pipeline([
                ('selector', NumberSelector(key='harsh_count')),
                ('standard', StandardScaler()),
            ])
            feature_list.append(('harsh_words', harsh_count))

        if self.positive_prop:
            print("Extracting proportion of positive words")
            positive_prop =  Pipeline([
                ('selector', NumberSelector(key='positive_proportion')),
                ('standard', StandardScaler()),
            ])
            feature_list.append(('positive_proportion', positive_prop))

        if self.negative_prop:
            print("Extracting proportion of negative words")
            negative_prop =  Pipeline([
                ('selector', NumberSelector(key='negative_proportion')),
                ('standard', StandardScaler()),
            ])
            feature_list.append(('negative_proportion', negative_prop))

        if self.harsh_prop:
            print("Extracting proportion of harsh words")
            harsh_prop =  Pipeline([
                ('selector', NumberSelector(key='harsh_proportion')),
                ('standard', StandardScaler()),
            ])
            feature_list.append(('harsh_proportion', harsh_prop))

        if self.word_ngram:
            print("Extracting word n-gram")
            word_ngram = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('word_ngram', CountVectorizer(ngram_range=(1, 1))),
            ])
            feature_list.append(('word_ngram', word_ngram))

        if self.char_ngram:
            print("Extracting char n-gram")
            char_ngram = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('char_ngram', CountVectorizer(ngram_range=(4, 4), analyzer='char')),
            ])
            feature_list.append(('char_ngram', char_ngram))

        self.feats = FeatureUnion(feature_list)

    def construct_classifier(self):
        if self.classifier_name == 'svm':
            print("Constructing SVM model")
            self.classifier = LinearSVC()
        elif self.classifier_name == 'log_reg':
            print("Constructing Logistic Regression model")
            self.classifier = linear_model.LogisticRegression()
        elif self.classifier_name == 'rfdt':
            print("Constructing RFDT model")
            self.classifier = RandomForestClassifier(random_state=3)

    def construct_pipeline(self):
        self.pipeline = Pipeline([
            ('features', self.feats),
            ('classifier', self.classifier),
        ])
        self.pipeline.fit(self.X_train, self.y_train)

    def split_data(self, test_size=0.1, random_state=42):
        print(self.df.label.value_counts())
        self.features= [c for c in self.df.columns.values if c  not in []]
        self.target = 'label'

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.features], 
                                                                                self.df[self.target],
                                                                                test_size=test_size,
                                                                                random_state=random_state)
    def train_data(self):
        print("Train data: cross validation =", self.k_cross_val)
        self.scores_f1 = cross_val_score(self.pipeline,
                                                  self.X_train, 
                                                  self.y_train, 
                                                  cv=self.k_cross_val, 
                                                  scoring='f1_weighted')
        self.mean_score = self.scores_f1.mean()
        print("F1 Weighted:", self.mean_score)

        self.y_pred = cross_val_predict(self.pipeline, 
                                        self.X_train, 
                                        self.y_train, 
                                        cv=self.k_cross_val)

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_train, self.y_pred).ravel()
        # tn, fp, fn, tp, precision, recall, f1
        self.precision = self.tp/(self.tp+self.fp)
        self.recall = self.tp/(self.tp+self.fn)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        print("Confusion_matrix:")
        print("TN:", self.tn)
        print("FP:", self.fp)
        print("FN:", self.fn)
        print("TP:", self.tp)
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        print("F1:", self.f1)

    def test_data(self):
        print("Test data")
        self.preds = self.pipeline.predict(self.X_test)
        print("Accuracy test:", np.mean(self.preds == self.y_test))
        self.tn_test, self.fp_test, self.fn_test, self.tp_test = confusion_matrix(self.y_test, self.preds).ravel()
        print("Confusion_matrix test:")
        print("TN:", self.tn_test)
        print("FP:", self.fp_test)
        print("FN:", self.fn_test)
        print("TP:", self.tp_test)
        
    def run(self):
        self.load_stopwords()
        self.load_positive_words()
        self.load_negative_words()
        self.load_positive_words('data/InSet-Lexicon/positive.tsv')
        self.load_negative_words('data/InSet-Lexicon/negative.tsv')
        self.load_negative_words('data/sentiLexiconNeg.txt')
        self.load_harsh_words()
        self.load_kamus_alay()
        self.load_typo_dict()
        self.load_data()
        self.processing()
        # self.data_balancing()
        self.split_data()
        self.extract_features()
        self.construct_classifier()
        self.construct_pipeline()
        self.train_data() 
        self.test_data()    
