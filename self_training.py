#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy 
import time
import glob
import logging
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
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.datasets import load_breast_cancer
from modules.data_preprocessing import DataPreprocessing
warnings.filterwarnings("ignore")

class SelfTraining:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, use_unlabeled_pool, group, confidence, pool_size, k):
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
        self.base_classifier = eval(self.setting_dict[group])
        self.threshold_dict = {}
        self.confidence = confidence
        if self.use_unlabeled_pool:
            self.unlabeled_pool = []
            self.create_pool()
        else:
            #if not pool then unlabeled_pool = all X_unlabeled
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
    
    def add_to_labeled(self, preds_one):
        preds = []
        for i in range(len(preds_one)):
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
    
    def train(self):
        bEnd = False
        bError = False
        for i in range(self.k):
            if bEnd:
                break
            #retrain classifier
            self.base_classifier = eval(self.setting_dict[self.group])
            self.base_classifier.fit(pd.get_dummies(self.X_labeled), self.y_labeled)
            #predict unlabeled data
            pred = self.base_classifier.predict_proba(pd.get_dummies(self.unlabeled_pool))
            #choose n unlabeled data which have the biggest confidence
            preds = []
            for j in range(pred.shape[1]):
                n_big = 0
                if self.n[j] > len(self.unlabeled_pool):
                    n_big = len(self.unlabeled_pool) - 1
                else:
                    n_big = self.n[j]
                temp = np.argpartition(pred[:, j], n_big * -1)[n_big * -1:]
                temp = temp.tolist()
                remove_list = []
                for ele in temp:
                    if pred[ele, j] < self.confidence:
                        remove_list.append(ele)
                for ele in remove_list:
                    temp.remove(ele)
                preds.append(temp)
            #add unlabeled data to labeled data
            bEnd = self.add_to_labeled(preds)
        self.unlabeled_num = len(self.X_labeled) - self.ori_X_labeled_ln 
        return bError
    
    def evaluate(self, X_test, y_test, avg = None):
        y_pred = self.base_classifier.predict(pd.get_dummies(X_test))
        y_test = y_test.tolist()
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average = avg)
        return precision, recall, fscore, support, y_pred, self.unlabeled_num

if __name__ == "__main__":
    #algorithm parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--groups", type = str, nargs = "+",
                        choices = ["NB", "SVM", "RF", "AdaBoost", "KNN", "DT"],
                        default = ["NB"],                        
                        help = "the type of base classifier")
    parser.add_argument("-c", "--confidence_thresholds", type = float, nargs = "+",
                        default = [0.1],
                        help = "the confidence threshold which decide whether to add unlabeled data")
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
                        help = "number of iterations of self-training algorithm")
    #data and log parameters
    parser.add_argument("--data_pre_type", type = str,
                        choices = ["all_category", "category_and_numeric"],
                        default = "category_and_numeric",
                        help = """
                        category => 10 equal-width bins
                        numeric => log
                        """)
    parser.add_argument("--data_dir", type = str,
                        default = None,
                        help = """
                        if data_dir == None: read data from sklearn.datasets package
                        else: read csv data from data directory
                        """)
    parser.add_argument("--log_dir", type = str,
                        default = os.path.join(os.getcwd(), "log", "self_training_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())),
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
    os.mkdir(args.log_dir)
    data_pre = DataPreprocessing()
    if args.data_dir == None:
        dataset = load_breast_cancer()
        X = dataset["data"]
        y = dataset["target"]
        X = pd.DataFrame(X)
        y = pd.Series(y)
        if args.data_pre_type == "category_and_numeric":
            df = data_pre.category_and_numeric(X)
        elif args.data_pre_type == "all_category":
            df = data_pre.columns_category(X)
        else:
            pass
        y = data_pre.to_category(y)
        if args.save_pred:
            log_dict = {"dataset":[],"group":[],"use_unlabeled_pool":[],"label_rate":[],
                    "avg_type":[],"precision":[],"recall":[],"fscore":[],"confidence_threshold":[],"y_pred":[],"y_test":[],
                    "unlabeled_num":[]}
        else:
            log_dict = {"dataset":[],"group":[],"use_unlabeled_pool":[],"label_rate":[],
                    "avg_type":[],"precision":[],"recall":[],"fscore":[],"confidence_threshold":[]}
        for ele in range(args.experiment_num):
            print("experiment_num : " + str(ele))
            for confidence_threshold in args.confidence_thresholds:
                for label_rate in args.label_rates:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = args.train_size, stratify = y, random_state = ele) 
                    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size = label_rate, stratify = y_train, random_state = ele)
                    delta_one_avg = 0
                    delta_two_avg = 0
                    for group in args.groups:
                        print("group : " +  group)
                        self_training = SelfTraining(X_labeled, y_labeled, X_unlabeled, args.use_unlabeled_pool, group, confidence_threshold, args.pool_size, args.k)
                        self_training.train()
                        for avg_type in args.avg_types:
                            precision, recall, fscore, _, y_pred, unlabeled_num = self_training.evaluate(X_test, y_test, avg_type)
                            log_dict["dataset"].append("breast_cancer_w")
                            log_dict["group"].append(group)
                            log_dict["use_unlabeled_pool"].append(args.use_unlabeled_pool)
                            log_dict["label_rate"].append(label_rate)
                            log_dict["avg_type"].append(avg_type)
                            log_dict["precision"].append(precision)
                            log_dict["recall"].append(recall)
                            log_dict["fscore"].append(fscore)
                            log_dict["confidence_threshold"].append(confidence_threshold)
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
            if args.data_pre_type == "category_and_numeric":
                df = data_pre.category_and_numeric(df)
            elif args.data_pre_type == "all_category":
                df = data_pre.columns_category(df)
            else:
                print("data_pre_type error!")
                continue
            X = df[df.columns[:-1]]
            y = df[df.columns[-1]]
            if args.save_pred:
                log_dict = {"dataset":[],"group":[],"use_unlabeled_pool":[],"label_rate":[],
                        "avg_type":[],"precision":[],"recall":[],"fscore":[],"confidence_threshold":[],"y_pred":[],"y_test":[],
                        "unlabeled_num":[]}
            else:
                log_dict = {"dataset":[],"group":[],"use_unlabeled_pool":[],"label_rate":[],
                        "avg_type":[],"precision":[],"recall":[],"fscore":[],"confidence_threshold":[]}
            for ele in range(args.experiment_num):
                print("experiment_num : " + str(ele))
                for confidence in args.confidences:
                    for label_rate in args.label_rates:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = args.train_size, stratify = y, random_state = ele) 
                        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size = label_rate, stratify = y_train, random_state = ele)
                        delta_one_avg = 0
                        delta_two_avg = 0
                        for group in args.groups:
                            print("group : " + group)
                            self_training = SelfTraining(X_labeled, y_labeled, X_unlabeled, args.use_unlabeled_pool, group, confidence, args.pool_size, args.k)
                            self_training.train()
                            for avg_type in args.avg_types:
                                precision, recall, fscore, _, y_pred, unlabeled_num = self_training.evaluate(X_test, y_test, avg_type)
                                log_dict["dataset"].append(os.path.basename(data))
                                log_dict["group"].append(group)
                                log_dict["use_unlabeled_pool"].append(args.use_unlabeled_pool)
                                log_dict["label_rate"].append(label_rate)
                                log_dict["avg_type"].append(avg_type)
                                log_dict["precision"].append(precision)
                                log_dict["recall"].append(recall)
                                log_dict["fscore"].append(fscore)
                                log_dict["confidence_threshold"].append(confidence)
                                if args.save_pred:
                                    log_dict["y_pred"].append(y_pred)
                                    log_dict["y_test"].append(y_test.tolist())
                                    log_dict["unlabeled_num"].append(unlabeled_num)
            df_log = pd.DataFrame(log_dict)
            df_log.to_csv(os.path.join(args.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + os.path.basename(data)), index = False)
    print("finish!")










