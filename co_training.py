#!/usr/bin/env python
# coding: utf-8

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
from tqdm import tqdm
import warnings
import gc
import argparse

from sklearn import tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import sys
from modules.data_preprocessing import DataPreprocessing
warnings.filterwarnings("ignore")

class ViewSplit():
    def __init__(self, group, cv = 3):
        self.setting_dict = {
        "NB" : "naive_bayes.GaussianNB()",
        "AdaBoost" : "AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.6,n_estimators=30, random_state=0)",
        "SVM" : """make_pipeline(StandardScaler(),SVC(C=0.6, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
            max_iter=100, probability=True,random_state=None, shrinking=True,
            tol=0.001, verbose=False))""",
        "RF" : """RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                       criterion='gini', max_depth=None, max_features=6,
                                       max_leaf_nodes=None, max_samples=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=60,
                                       n_jobs=None, oob_score=False, random_state=None,
                                       verbose=0, warm_start=False)""",
        "DT" : "DecisionTreeClassifier(criterion = 'entropy')",
        "KNN" : "KNeighborsClassifier(n_neighbors = 10)"
        }
        self.cv = cv
        self.base_classifier = eval(self.setting_dict[group[0]])
        
    def compute_majority_category(self, column):
        check_dict = {}
        for row in column:
            temp = check_dict.get(row)
            if temp == None:
                check_dict.update({row:1})
            else:
                check_dict[row] += 1
        majority_category = max(check_dict.items(),key = lambda ele:ele[1])
        proportion = majority_category[1] / len(column)
        return majority_category, proportion
    
    def compute_delta_one(self, df, view_one, view_two):
        """
        sufficiency parameter
        find the small positive delta_one such as:
        p > 1 - delta_one,
        px > 1 - delta_one,
        py > 1 - delta_one.
        """
        view_all = view_one[:]
        view_all.extend(view_two)
        X = pd.get_dummies(df[view_all])
        y = df[df.columns[-1]]
        p_scores = cross_val_score(self.base_classifier, X, y, cv = self.cv, scoring = "accuracy")
        X = pd.get_dummies(df[view_one])
        y = df[df.columns[-1]]
        px_scores = cross_val_score(self.base_classifier, X, y, cv = self.cv, scoring = "accuracy")
        X = pd.get_dummies(df[view_two])
        y = df[df.columns[-1]]
        py_scores = cross_val_score(self.base_classifier, X, y, cv = self.cv, scoring = "accuracy")
        scores = [sum(p_scores) / len(p_scores), sum(px_scores) / len(px_scores), sum(py_scores) / len(py_scores)]
        delta_one = 1 - min(scores)
        return delta_one
        
    def compute_delta_two(self, df, view_one, view_two):
        """
        independence parameter
        find the small positive delta_two such as:
        pxi < p'xi + delta_two for all 1 <= i <= m, and
        pyi < p'yi + delta_two for all 1 <= i <= n.
        """
        X = pd.get_dummies(df[view_one])
        delta_two = 0.0
        for feature in view_two:
            y = df[feature]
            scores = cross_val_score(self.base_classifier, X, y, cv = self.cv, scoring = "accuracy")
            score = sum(scores) / len(scores)
            _, proportion = self.compute_majority_category(df[feature])
            if (score - proportion) > delta_two:
                delta_two = (score - proportion)

        X = pd.get_dummies(df[view_two])
        for feature in view_one:
            y = df[feature]
            scores = cross_val_score(self.base_classifier, X, y, cv = self.cv, scoring = "accuracy")
            score = sum(scores) / len(scores)
            _, proportion = self.compute_majority_category(df[feature])
            if (score - proportion) > delta_two:
                delta_two = (score - proportion)
        return delta_two
    
    def compute_delta(self, df, view_one, view_two):
        delta_one = self.compute_delta_one(df, view_one, view_two)
        delta_two = self.compute_delta_two(df, view_one, view_two)
        return delta_one, delta_two
    
    def random_split(self, df):
        #random split single view into two views
        indexs = np.random.choice(df.columns[:-1], len(df.columns[:-1]), replace = False)
        view_one = indexs[:int(len(df.columns[:-1]) / 2)]
        view_two = indexs[int(len(df.columns[:-1]) / 2):]
        view_one = view_one.tolist()
        view_two = view_two.tolist()
        return view_one, view_two
    
    def compute_entropy(self, df):
        entropy_dict = {}
        nodes_entropy = []
        for feature in df.columns[:-1]:
            categories_entropy = []
            categories_num = []
            for category in df[feature].cat.categories:
                tags_num = []
                tags = df.columns[-1]
                for tag in df[tags].cat.categories:
                    is_category = df[feature] == category
                    is_tag = df[tags] == tag
                    tags_num.append(len(df[is_category & is_tag]))
                tags_num = np.array(tags_num)
                tags_num = tags_num / sum(tags_num)
                categories_entropy.append(entropy(tags_num, base = 2))
                categories_num.append(len(df[df[feature] == category]))
            categories_entropy = np.array(categories_entropy)
            categories_num = np.array(categories_num)
            categories_num = categories_num / sum(categories_num)
            category_entropy = sum(categories_entropy * categories_num)
            entropy_dict.update({feature : category_entropy})
        entropy_dict = dict(sorted(entropy_dict.items(),key = lambda ele : ele[1]))
        return entropy_dict
            
    def entropy_split(self, df):
        """
        compute entropy for each feature
        sort features by entropy value
        assign odd order feature to view one, even order feature to view two
        """
        entropy_dict = self.compute_entropy(df)
        view_one = []
        view_two = []
        i = 0
        for key in entropy_dict.keys():
            if i % 2 == 0:
                view_one.append(key)
            else:
                view_two.append(key)
            i += 1
        return view_one, view_two
        
    def change_view(self, view_one, view_two):
        views = []
        for i in range(len(view_one)):
            for j in range(len(view_two)):
                view_one_new = view_one[:]
                view_two_new = view_two[:]
                temp = view_one_new[i]
                view_one_new[i] = view_two_new[j]
                view_two_new[j] = temp
                views.append([view_one_new, view_two_new])
        return views
    
    def entropy_hill(self, df):
        #use entropy_split to split single view into two views
        #compute delta one and delta two for each view pairs
        #use the hill climbing algorithm to find a view pairs which have the smallest delta one + delta two
        t0 = time.time()
        print("entropy_hill start")
        min_view_one, min_view_two = self.entropy_split(df)
        min_delta_one, min_delta_two = self.compute_delta(df, min_view_one, min_view_two)
        iRun = 0
        while True:
            if iRun % 1 == 0:
                print("turn: " + str(iRun))
            views = self.change_view(min_view_one, min_view_two)
            bChange = False
            for view in views:
                delta_one, delta_two = self.compute_delta(df, view[0], view[1])
                if (delta_one + delta_two) < (min_delta_one + min_delta_two):
                    min_view_one, min_view_two = view[0], view[1]
                    min_delta_one, min_delta_two = delta_one, delta_two
                    bChange = True
            if not bChange:
                break
            iRun += 1
        print(time.time() - t0)
        print("entropy_hill end")
        return min_view_one, min_view_two
            
    def random_hill(self, df):
        #use random_split to split single view into two views
        #compute delta one and delta two for each view pairs
        #use the hill climbing algorithm to find a view pairs which have the smallest delta one + delta two
        print("random_hill start")
        t0 = time.time()
        splits = []
        for i in range(20):
            splits.append(self.random_split(df))
        views = []
        for i in range(len(splits)):
            split = splits[i]
            views.extend(self.change_view(split[0], split[1]))
        views.extend(splits)
        min_view_one = []
        min_view_two = []
        min_delta_one = 99999999
        min_delta_two = 99999999
        bFR = True
        iRun = 0
        while True:
            if iRun % 1 == 0:
                print("turn: " + str(iRun))
            t1 = time.time()
            bChange = False
            if bFR:
                bFR = False
            else:
                views = self.change_view(min_view_one, min_view_two)
            for view in views:
                delta_one, delta_two = self.compute_delta(df, view[0], view[1])
                if (delta_one + delta_two) < (min_delta_one + min_delta_two):
                    min_view_one, min_view_two = view[0], view[1]
                    min_delta_one, min_delta_two = delta_one, delta_two
                    bChange = True
            print(time.time() - t1)
            if not bChange:
                break
            iRun += 1
        print(time.time() - t0)
        print("random_hill end")
        return min_view_one, min_view_two        

class CoTraining:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, use_unlabeled_pool, group, pool_size, k):
        self.setting_dict = {
        "NB" : "naive_bayes.GaussianNB()",
        "AdaBoost" : "AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.6,n_estimators=30, random_state=0)",
        "SVM" : """make_pipeline(StandardScaler(),SVC(C=0.6, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
            max_iter=100, probability=True,random_state=None, shrinking=True,
            tol=0.001, verbose=False))""",
        "RF" : """RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                       criterion='gini', max_depth=None, max_features=6,
                                       max_leaf_nodes=None, max_samples=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=60,
                                       n_jobs=None, oob_score=False, random_state=None,
                                       verbose=0, warm_start=False)""",
        "DT" : "DecisionTreeClassifier(criterion = 'entropy')",
        "KNN" : "KNeighborsClassifier(n_neighbors = 10)"
        }
        self.pool_size = pool_size
        self.k = k
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.ori_X_labeled_ln = len(X_labeled)
        self.ori_y_labeled_ln = len(y_labeled)
        self.unlabeled_num = 0
        self.n = self.count_n()
        self.X_unlabeled = X_unlabeled
        self.use_unlabeled_pool = use_unlabeled_pool
        self.group = group
        self.base_classifier_one = eval(self.setting_dict[group[0]])
        self.base_classifier_two = eval(self.setting_dict[group[1]])
        if self.use_unlabeled_pool:
            self.unlabeled_pool = []
            self.create_pool()
        else:
            #if not pool:unlabeled_pool = all X_unlabeled
            self.unlabeled_pool = self.X_unlabeled.copy()
        
    def count_n(self):
        result = []
        class_dict = {}
        for i in range(len(self.y_labeled)):
            temp = class_dict.get(self.y_labeled.iloc[i])
            if temp == None:
                class_dict.update({self.y_labeled.iloc[i]:1})
            else:
                class_dict[self.y_labeled.iloc[i]] += 1
        min_value = 999999
        for key, value in class_dict.items():
            if value < min_value:
                min_value = value
        for key, value in class_dict.items():
            temp = int(value / min_value)
            if temp < 1:
                temp = 1
            result.append(temp)
        return result
            
    def create_pool(self):
        if self.pool_size < len(self.X_unlabeled):
            indexs = np.random.choice(self.X_unlabeled.index, self.pool_size, replace = False)
            self.unlabeled_pool = self.X_unlabeled.loc[indexs,:]
            self.X_unlabeled = self.X_unlabeled.drop(index = indexs)
        else:
            print("dataset too small can't create pool!")
        
    def check_pred_duplicate(self, preds):
        remove_list = []
        result = []
        for cla in preds:
            result.append([])
        for i in range(len(preds)):
            for j in range(i + 1,len(preds)):
                for k in range(len(preds[i])):
                    if preds[i][k] in preds[j]:
                        remove_list.append(preds[i][k])
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                if preds[i][j] not in remove_list:
                    result[i].append(preds[i][j])
        return result
    
    def add_to_labeled(self, preds_one, preds_two):
        #if the data in preds_one and preds_two: drop the data
        preds = []
        for i in range(len(preds_one)):
            preds_one[i].extend(preds_two[i])
            preds.append(list(set(preds_one[i])))
        preds = self.check_pred_duplicate(preds)
        drops = []
        for i in range(len(preds)):
            if len(preds[i]) > 0:
                pred_y = pd.Series([self.y_labeled.cat.categories[i]] * len(preds[i]), 
                                     index = self.unlabeled_pool.iloc[preds[i],:].index,
                                     dtype = "category")
                pred_y.cat.set_categories(self.y_labeled.cat.categories, inplace = True)
                self.X_labeled = self.X_labeled.append(self.unlabeled_pool.iloc[preds[i],:])
                self.y_labeled = self.y_labeled.append(pred_y)
                drops.append(pred_y)
        for i in range(len(drops)):
            self.unlabeled_pool = self.unlabeled_pool.drop(index = drops[i].index)
        if self.use_unlabeled_pool:
            for i in range(len(drops)):
                self.replenish_pool(len(drops[i]))
        if len(self.unlabeled_pool) < 1 or len(drops) < 1:
            bEnd = True
        else:
            bEnd = False
        return bEnd
    
    def replenish_pool(self, sample_size):
        if sample_size > len(self.X_unlabeled):
            sample_size = len(self.X_unlabeled)
        if sample_size > 0:
            indexs = np.random.choice(self.X_unlabeled.index, sample_size, replace = False)
            self.unlabeled_pool = self.unlabeled_pool.append(self.X_unlabeled.loc[indexs,:])
            self.X_unlabeled = self.X_unlabeled.drop(index = indexs)
    
    def train(self, view_one, view_two):
        bEnd = False
        bError = False
        for i in range(self.k):
            if bEnd:
                break
            #retrain calssifiers
            #classifier one only use the view_one
            #classifier two only use the view_two
            self.base_classifier_one = eval(self.setting_dict[self.group[0]])
            self.base_classifier_two = eval(self.setting_dict[self.group[1]])
            self.base_classifier_one.fit(pd.get_dummies(self.X_labeled[view_one]), self.y_labeled)
            self.base_classifier_two.fit(pd.get_dummies(self.X_labeled[view_two]), self.y_labeled)
            #preict unlabeled data
            pred_one = self.base_classifier_one.predict_proba(pd.get_dummies(self.unlabeled_pool[view_one]))
            pred_two = self.base_classifier_two.predict_proba(pd.get_dummies(self.unlabeled_pool[view_two]))
            #choose n unlabeled data which have the biggest confidence
            preds_one = []
            preds_two = []
            for j in range(pred_one.shape[1]):
                n_big = 0
                if self.n[j] > len(self.unlabeled_pool):
                    n_big = len(self.unlabeled_pool) - 1
                else:
                    n_big = self.n[j]
                temp = np.argpartition(pred_one[:, j], n_big * -1)[n_big * -1:]
                temp = temp.tolist()
                preds_one.append(temp)
            for j in range(pred_two.shape[1]):
                n_big = 0
                if self.n[j] > len(self.unlabeled_pool):
                    n_big = len(self.unlabeled_pool) - 1
                else:
                    n_big = self.n[j]
                temp = np.argpartition(pred_two[:, j], n_big * -1)[n_big * -1:]
                temp = temp.tolist()
                preds_two.append(temp)
            #add unlabeled data to labeled data
            bEnd = self.add_to_labeled(preds_one, preds_two)
        self.unlabeled_num = len(self.X_labeled) - self.ori_X_labeled_ln 
        return bError
    
    def evaluate(self, X_test, y_test, view_one, view_two, avg = None):
        #when co-trainig predict test data, it will product the probabilities of two classifiers
        pred_one = self.base_classifier_one.predict_proba(pd.get_dummies(X_test[view_one]))
        pred_two = self.base_classifier_two.predict_proba(pd.get_dummies(X_test[view_two]))
        pred_co_training = pred_one * pred_two
        y_pred = []
        for row in pred_co_training:
            y_pred.append(y_test.cat.categories[np.argmax(row)])
        y_test = y_test.tolist()
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average = avg)
        return precision, recall, fscore, support, y_pred, self.unlabeled_num

if __name__ == "__main__":
    #algorithm parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_types", type = str, nargs = "+",
                        choices = ["random_split", "entropy_split", "entropy_hill", "random_hill"],
                        default = ["entropy_split"],
                        help = "the method which split the single view(feature subset) into two views")
    parser.add_argument("--split_small",
                        action = "store_true",
                        help = "whether split view only based on labeled data or whole dataset")  
    parser.add_argument("-g", "--groups", type = str, nargs = "+",
                        choices = ["NB", "SVM", "RF", "AdaBoost", "KNN", "DT"],
                        default = ["NB"],                        
                        help = "the two base classifiers use the same type of classifier")
    parser.add_argument("-t", "--train_size", type = float,
                        default = 0.75,
                        help = "the size of the training dataset")
    parser.add_argument("-l", "--label_rates", type = float, nargs = "+",
                        default = [0.1],
                        help = "the size of the labeled data = train_size * label_rate")
    parser.add_argument("-e", "--experiment_num", type = int,
                        default = 10,
                        help = "number of experiments")
    parser.add_argument("--use_unlabeled_pool",
                        action = "store_true",
                        help = "whether to use the unlabeled pool")
    parser.add_argument("--pool_size", type = int,
                        default = 75,
                        help = "the size of the unlabeled pool")
    parser.add_argument("-k", type = int,
                        default = 30,
                        help = "number of iterations of co-training algorithm")
    #data and log parameters
    parser.add_argument("--data_pre_type", type = str,
                        choices = ["all_category"],
                        default = "all_category",
                        help = "category => 10 equal-width bins")
    parser.add_argument("--data_dir", type = str,
                        default = None,
                        help = """
                        if data_dir == None: read data from sklearn.datasets package
                        else: read csv data from data directory
                        """)
    parser.add_argument("--log_dir", type = str,
                        default = os.path.join(os.getcwd(), "log", "co_training_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())),
                        help = "log directory")
    parser.add_argument("--avg_types", type = str, nargs = "+",
                        choices = ["micro", "macro", "weighted"],
                        default = ["weighted"],
                        help = """
                        micro => micro average
                        macro => macro average
                        weighted => weighted macro average
                        """)
    parser.add_argument("--save_pred", 
                        action = "store_true",
                        help = "whether to save prediction and number of unlabeled")
    args = parser.parse_args()
    for i in range(len(args.groups)):
        args.groups[i] = [args.groups[i]] * 2
    data_pre = DataPreprocessing()
    os.mkdir(args.log_dir)
    if args.data_dir == None:
        dataset = load_breast_cancer()
        X = dataset["data"]
        y = dataset["target"]
        X = pd.DataFrame(X)
        y = pd.Series(y)
        if args.data_pre_type == "all_category":
            X = data_pre.columns_category(X)
        else:
            print("data_pre_type error!")
        y = data_pre.to_category(y)
        if args.save_pred:
            log_dict = {"dataset":[],"group":[],"split_small":[],"split_type":[],"view_one":[],"view_two":[],
                        "use_unlabeled_pool":[],"label_rate":[],"delta_one":[],"delta_two":[],
                        "avg_type":[],"precision":[],"recall":[],"fscore":[],"y_pred":[],"y_test":[],
                        "unlabeled_num":[]}
        else:
            log_dict = {"dataset":[],"group":[],"split_small":[],"split_type":[],"view_one":[],"view_two":[],
                        "use_unlabeled_pool":[],"label_rate":[],"delta_one":[],"delta_two":[],
                        "avg_type":[],"precision":[],"recall":[],"fscore":[]}
        for ele in range(args.experiment_num):
            print("experiment_num : " + str(ele))
            for label_rate in args.label_rates:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = args.train_size, stratify = y, random_state = ele) 
                X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size = label_rate, stratify = y_train, random_state = ele)
                df = X.assign(new = y)
                df_labeled = X_labeled.assign(new = y_labeled)
                delta_one_avg = 0
                delta_two_avg = 0
                for group in args.groups:
                    print("group : " + "_".join(group))
                    view_split = ViewSplit(group)
                    for split_type in args.split_types:
                        if args.split_small:
                            view_one, view_two = eval("view_split.{}(df_labeled)".format(split_type))
                        else:
                            view_one, view_two = eval("view_split.{}(df)".format(split_type))
                        delta_one, delta_two = view_split.compute_delta(df, view_one, view_two)
                        co_training = CoTraining(X_labeled, y_labeled, X_unlabeled, args.use_unlabeled_pool, group, args.pool_size, args.k)
                        co_training.train(view_one, view_two)
                        for avg_type in args.avg_types:
                            precision, recall, fscore, _, y_pred, unlabeled_num = co_training.evaluate(X_test, y_test, view_one, view_two, avg_type)
                            log_dict["dataset"].append("breast_cancer_w")
                            log_dict["group"].append("_".join(group))
                            log_dict["split_small"].append(args.split_small)
                            log_dict["split_type"].append(split_type)
                            log_dict["view_one"].append(sorted(view_one))
                            log_dict["view_two"].append(sorted(view_two))
                            log_dict["use_unlabeled_pool"].append(args.use_unlabeled_pool)
                            log_dict["label_rate"].append(label_rate)
                            log_dict["delta_one"].append(delta_one)
                            log_dict["delta_two"].append(delta_two)
                            log_dict["avg_type"].append(avg_type)
                            log_dict["precision"].append(precision)
                            log_dict["recall"].append(recall)
                            log_dict["fscore"].append(fscore)
                            if args.save_pred:
                                log_dict["y_pred"].append(y_pred)
                                log_dict["y_test"].append(y_test.tolist())
                                log_dict["unlabeled_num"].append(unlabeled_num)
        df_log = pd.DataFrame(log_dict)
        df_log.to_csv(os.path.join(args.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + "breast_cancer_w.csv"), index = False)
    else:
        for data in glob.glob(os.path.join(args.data_dir, "*.csv")):
            print(os.path.basename(data))
            df = pd.read_csv(data)
            if args.data_pre_type == "all_category":
                df = data_pre.columns_category(df)
            else:
                print("data_pre_type error!")
                continue
            X = df[df.columns[:-1]]
            y = df[df.columns[-1]]
            if args.save_pred:
                log_dict = {"dataset":[],"group":[],"split_small":[],"split_type":[],"view_one":[],"view_two":[],
                            "use_unlabeled_pool":[],"label_rate":[],"delta_one":[],"delta_two":[],
                            "avg_type":[],"precision":[],"recall":[],"fscore":[],"y_pred":[],"y_test":[],
                            "unlabeled_num":[]}
            else:
                log_dict = {"dataset":[],"group":[],"split_small":[],"split_type":[],"view_one":[],"view_two":[],
                            "use_unlabeled_pool":[],"label_rate":[],"delta_one":[],"delta_two":[],
                            "avg_type":[],"precision":[],"recall":[],"fscore":[]}
            for ele in range(args.experiment_num):
                print("experiment_num : " + str(ele))
                for label_rate in args.label_rates:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = args.train_size, stratify = y, random_state = ele) 
                    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size = label_rate, stratify = y_train, random_state = ele)
                    df_labeled = X_labeled.assign(new = y_labeled)
                    delta_one_avg = 0
                    delta_two_avg = 0
                    for group in args.groups:
                        print("group : " + "_".join(group))
                        view_split = ViewSplit(group)
                        for split_type in args.split_types:
                            if args.split_small:
                                view_one, view_two = eval("view_split.{}(df_labeled)".format(split_type))
                            else:
                                view_one, view_two = eval("view_split.{}(df)".format(split_type))
                            delta_one, delta_two = view_split.compute_delta(df, view_one, view_two)
                            co_training = CoTraining(X_labeled, y_labeled, X_unlabeled, args.use_unlabeled_pool, group, args.pool_size, args.k)
                            co_training.train(view_one, view_two)
                            for avg_type in args.avg_types:
                                precision, recall, fscore, _, y_pred, unlabeled_num = co_training.evaluate(X_test, y_test, view_one, view_two, avg_type)
                                log_dict["dataset"].append(os.path.basename(data))
                                log_dict["group"].append("_".join(group))
                                log_dict["split_small"].append(args.split_small)
                                log_dict["split_type"].append(split_type)
                                log_dict["view_one"].append(sorted(view_one))
                                log_dict["view_two"].append(sorted(view_two))
                                log_dict["use_unlabeled_pool"].append(args.use_unlabeled_pool)
                                log_dict["label_rate"].append(label_rate)
                                log_dict["delta_one"].append(delta_one)
                                log_dict["delta_two"].append(delta_two)
                                log_dict["avg_type"].append(avg_type)
                                log_dict["precision"].append(precision)
                                log_dict["recall"].append(recall)
                                log_dict["fscore"].append(fscore)
                                if args.save_pred:
                                    log_dict["y_pred"].append(y_pred)
                                    log_dict["y_test"].append(y_test.tolist())
                                    log_dict["unlabeled_num"].append(unlabeled_num)
            df_log = pd.DataFrame(log_dict)
            df_log.to_csv(os.path.join(args.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + os.path.basename(data)), index = False)
    print("finish!")

