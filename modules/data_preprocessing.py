#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
import time
import glob
import logging
from tqdm import tqdm
import warnings
import gc
from sklearn import tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier, Pool
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class DataPreprocessing:
    def __init__(self, equal_width_bins_num = 10, max_classes_num = 10):
        self.equal_width_bins_num = equal_width_bins_num
        self.max_classes_num = max_classes_num
        self.numerics = ["float16", "float64"]
        self.categories = ["int16", "int64", "category", "object"]
    
    def is_numeric(self, column):
        if column.dtype in self.numerics:
            return True
        else:
            return False
    
    def multi_classes_to_n(self, column):
        #multiple classes to n classes
        column = column.astype("category")
        check_dict = {}
        new_categories = []
        for row in column:
            temp = check_dict.get(row)
            if temp == None:
                check_dict.update({row:1})
            else:
                check_dict[row] += 1
        n = min(self.max_classes_num, len(check_dict))
        sorted_class = sorted(check_dict.items(),key = lambda ele:ele[1],reverse = True)
        new_categories = []
        for i in range(n - 1):
            new_categories.append(sorted_class[i][0]) 
        old_categories = new_categories.copy()
        new_categories.append(sorted_class[n - 1][0])
        column.cat.set_categories(new_categories, inplace = True)
        for i in range(len(column)):
            if column.iloc[i] not in old_categories:
                column.iloc[i] = sorted_class[n - 1][0]
        return column
    
    def to_category(self,column):
        #10 equal-width bins
        if self.is_numeric(column):
            column = pd.cut(column, bins = self.equal_width_bins_num, labels = False)
        column = column.astype("category")
        return column

    def columns_category(self,df,to_binary = False):
        for column in df.columns:
            group_num = len(df[column].drop_duplicates())
            df[column] = self.to_category(df[column])
            if group_num > self.max_classes_num:
                df[column] = self.multi_classes_to_n(df[column])
        return df

    def to_numeric_logn(self, column, eplison = 0):
        #in order to avoid the negative values using log(x + 1)
        min_value = column.min()
        if min_value <= 0:
            add_value = abs(min_value) + eplison
            column = column + add_value
        column = np.log1p(column)
        return column

    def category_and_numeric(self, df):
        columns_is_category = []
        for column in df.columns:
            if not self.is_numeric(df[column]):
                df[column] = self.to_category(df[column])
                group_num = len(df[column].drop_duplicates())
                if group_num > self.max_classes_num:
                    df[column] = self.multi_classes_to_n(df[column])
                columns_is_category.append(column)
            else:
                df[column] = self.to_numeric_logn(df[column])
                df[column] = self.column_nomarize_max_min(df[column])
        return df

    def column_nomarize_max_min(self, column):
        column_norm = (column - column.min()) / (column.max() - column.min())
        return column_norm