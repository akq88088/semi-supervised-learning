#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import naive_bayes
from sklearn.datasets import load_breast_cancer
import datetime

class AutoSelectThreshold():
    def __init__(self):
        pass
        
    def auto_select_threshold_fit(self, model, probas, y_train = []):
        if len(y_train) > 0:
            class_indexs = list(y_train.cat.categories)
        else:
            class_indexs = list(model.classes_)
        threshold_dict = {}
        probas = np.array(probas)
        for index_name in class_indexs:
            yes_index = class_indexs.index(index_name)
            model_yes_proba = probas[:, yes_index]
            model_train = pd.DataFrame()
            #convert data to binary class
            correct_binary = self.column2binary(list(y_train), index_name)
            model_train['correct'] = correct_binary  
            model_train['yes_prob'] = model_yes_proba
            model_train.sort_values(by='yes_prob', ascending=False, inplace=True)  
            model_train.reset_index(drop=True, inplace=True)

            pred = [0] * model_train.shape[0]
            f1_prob_dict = dict()
            #compute f1-score for each f1-score
            for x in range(0, model_train.shape[0] - 1):
                pred[x] = 1
                f1 = round(f1_score(y_pred=pred, y_true=list(model_train['correct']), average='binary', pos_label=1), 3)
                this_prob = model_train['yes_prob'][x:x + 2].mean()
                f1_prob_dict[this_prob] = f1
            #select threshold which have the biggest f1-score
            f1_max = max(f1_prob_dict.values())
            prob_max = max(f1_prob_dict, key=f1_prob_dict.get)
            #threshold dict store best classification threshold for each class
            threshold_dict.update({index_name:prob_max})
        return threshold_dict

    def auto_select_threshold_predict(self, model, probas, y_test = [], threshold_dict = {}):
        class_index_dict = {}
        if len(y_test) >= 1:
            class_indexs = list(y_test.cat.categories)
        else:
            class_indexs = list(model.classes_)

        probas = np.array(probas)
        for index_name in class_indexs:
            yes_index = class_indexs.index(index_name)
            class_index_dict.update({yes_index:index_name})
            class_index_dict.update({index_name:yes_index})
            index_threshold = threshold_dict[index_name]
            probas[:, yes_index] = probas[:, yes_index] - index_threshold
        y_pred = []
        for i in range(len(probas)):
            #the class which have the biggest difference(probability - class threshold) will be selected
            max_class = self.select_argmax_category(probas[i,:], y_test)
            y_pred.append(class_index_dict[max_class])
        return y_pred

    def column2binary(self, column, index_name):
        for i in range(len(column)):
            if column[i] != index_name:
                column[i] = 0
            else:
                column[i] = 1
        return column

    def select_argmax_category(self, row, y):
        max_category = max(row)
        max_categories = []
        for i in range(len(row)):
            if row[i] == max_category:
                max_categories.append(i)
        #if the different is same then will compare the number of occurrences in the dataset
        if len(max_categories) > 1:
            category_num_dict = {}
            for i in range(len(y.cat.categories)):
                category_num = len(y[y == y.cat.categories[i]])
                category_num_dict.update({i : category_num})
            compare_list = []
            for category_index in max_categories:
                compare_list.append([category_index, category_num_dict[category_index]])
            return max(compare_list, key = lambda ele : ele[1])[0]
        else:
            return max_categories[0]

if __name__ == '__main__':
    auto_select = AutoSelectThreshold()
    dataset = load_breast_cancer()
    X = dataset["data"]
    y = dataset["target"]
    X = pd.DataFrame(X)
    y = pd.Series(y)
    y = y.astype("category")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 1) 
    model = naive_bayes.MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fscore = f1_score(y_pred = y_pred, y_true = list(y_test), average = 'macro') 
    print("before")
    print(fscore)
    probas_train = model.predict_proba(X_train)
    threshold_dict = auto_select.auto_select_threshold_fit(model = model, probas = probas_train, 
                                                           y_train = y_train)

    probas_test = model.predict_proba(X_test)
    y_pred = auto_select.auto_select_threshold_predict(model = model, probas = probas_test, y_test = y_test,
                                                       threshold_dict = threshold_dict)
    fscore = f1_score(y_pred = y_pred, y_true = list(y_test), average = 'macro') 
    print("after")
    print(fscore)

