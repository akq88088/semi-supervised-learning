#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support
import time
import datetime
import glob
from tqdm import tqdm
import warnings
from sklearn.utils import resample
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import math
from itertools import combinations
import random
import shutil
import argparse
warnings.filterwarnings("ignore")

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

import tensorflow as tf
import gc
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.per_process_gpu_memory_fraction = 0.7
from modules.data_preprocessing import DataPreprocessing
from modules.auto_select_threshold import AutoSelectThreshold

class TwoPhaseAutoFit:
    def __init__(self, base_classifier_str = [], X_train = [], y_train = [], X_unlabeled = [], data_pre_type = "", use_auto_select_threshold = False, confidence_threshold = None):
        self.setting_dict = {
        "NB" : "naive_bayes.GaussianNB()",
        "AdaBoost" : "AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.6,n_estimators=30, random_state=0)",
        "SVM" : """make_pipeline(StandardScaler(),SVC(C=0.6, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
            max_iter=100, probability=True,random_state=None, shrinking=True,
            tol=0.001, verbose=False))""",
        "RF" : """RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                       criterion='gini', max_depth=None,
                                       max_leaf_nodes=None, max_samples=None, 
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=60,
                                       n_jobs=None, oob_score=False, random_state=None,
                                       verbose=0, warm_start=False)""",
        "KNN" : "KNeighborsClassifier(n_neighbors = 10)",
        "DT" : "DecisionTreeClassifier(criterion = 'entropy')",
        "PCA" : "PCA(n_components=0.95, svd_solver = 'full')"
        }
        self.confidence_threshold = confidence_threshold
        self.use_auto_select_threshold = use_auto_select_threshold
        self.threshold_dict = {}
        self.auto_select = AutoSelectThreshold()
        self.data_pre = DataPreprocessing()
        self.data_pre_type = data_pre_type
        if base_classifier_str != []:
            self.base_classifiers_str = base_classifier_str
            self.feature_pre_dict = feature_pre_dict
            self.feature_pre_models = [0] * len(self.base_classifiers_str)
            self.homo_flag = self.compute_homo()
            self.X_labeled = X_train
            self.y_labeled = y_train
            self.unlabeled_num_sum = 0
            self.unlabeled_nums = [0] * len(self.base_classifiers_str)
            self.turn_nums = [0] * len(self.base_classifiers_str)
            self.X_unlabeled = X_unlabeled
            self.X_combine = []
            self.y_combine = []
            self.X_Si, self.y_Si = self.boostrap_sample()
            self.base_classifiers = self.init_classifier()
            self.second_model = [""] * len(self.base_classifiers)

    def compute_homo(self):
        homo_flag = [False] * len(self.base_classifiers_str)
        for i in range(len(self.base_classifiers_str)):
            for j in range(len(self.base_classifiers_str)):
                if j != i:
                    same_i = self.base_classifiers_str[i]
                    same_j = self.base_classifiers_str[j]
                    if same_i == same_j and self.feature_pre_dict[i] == self.feature_pre_dict[j]:
                        homo_flag[i] = True
                        break
        return homo_flag

    def init_classifier(self):
        result = []
        for i in range(len(self.base_classifiers_str)):
            if not self.homo_flag[i]:
                model = self.different_fit(i, self.X_labeled, self.y_labeled)
            else:
                model = self.different_fit(i, self.X_Si[i], self.y_Si[i])
            result.append(model)
        return result
    
    def boostrap_sample(self):
        X_Si = []
        y_Si = []
        for i in range(len(self.base_classifiers_str)):
            X_sample, y_sample = resample(self.X_labeled, self.y_labeled, stratify = self.y_labeled)
            X_Si.append(X_sample)
            y_Si.append(y_sample)
            self.X_combine.append(X_sample)
            self.y_combine.append(y_sample)
        return X_Si, y_Si

    def build_second_model(self, X, y):
        model = Sequential()
        model.add(Dense(8, input_shape=(len(X.columns),), activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(len(y.cat.categories), activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def second_model_fit(self, i, X, y, epochs = 30):
        y_one_hot = pd.get_dummies(y)
        self.second_model[i].fit(X, y_one_hot, epochs = epochs, verbose = 0)
        if self.use_auto_select_threshold:
            probas = self.second_model[i].predict_proba(X)
            self.threshold_dict = self.auto_select.auto_select_threshold_fit(self.second_model[i], probas, y)

    def remove_unlabeled_with_threshlod(self, probas):
        #if the confidence of data is less then the confidence_threshold: remove the data
        new_X_Li = []
        result = []
        probas = pd.DataFrame(np.array(probas))
        for i in range(len(probas)):
            temp = probas.iloc[i, :]
            max_class_confidence = temp.max()
            if max_class_confidence >= self.confidence_threshold:
                result.append(probas.iloc[i, :])
                new_X_Li.append(self.X_unlabeled.iloc[i, :])
            else:
                result.append(pd.Series([np.nan] * len(probas.iloc[i, :])))
                new_X_Li.append(pd.Series([np.nan] * len(self.X_unlabeled.iloc[i, :])))
        result = pd.DataFrame(np.array(result))
        new_X_Li = pd.DataFrame(np.array(new_X_Li))
        return result, new_X_Li

    def second_model_predict(self, i, X):
        if self.use_auto_select_threshold:
            probas = self.second_model[i].predict_proba(X)
            probas, new_X_Li = self.remove_unlabeled_with_threshlod(probas)
            probas = probas.dropna()
            new_X_Li = new_X_Li.dropna()
            probas = probas.to_numpy()
            new_X_Li = new_X_Li.to_numpy()
            y_pred = self.auto_select.auto_select_threshold_predict(self.second_model[i], probas, 
                                                                    y_test = self.y_labeled,
                                                                    threshold_dict = self.threshold_dict)
        else:
            y_pred = []
            class_index_dict = {}
            for ele in range(len(self.y_labeled.cat.categories)):
                class_index_dict.update({ele:self.y_labeled.cat.categories[ele]})
            probas = self.second_model[i].predict_proba(X)
            probas, new_X_Li = self.remove_unlabeled_with_threshlod(probas)
            probas = probas.dropna()
            new_X_Li = new_X_Li.dropna()
            for i in range(len(probas)):
                max_class = np.argmax(probas.iloc[i,:])
                y_pred.append(class_index_dict[max_class])
            new_X_Li = new_X_Li.to_numpy()
        return y_pred, new_X_Li

    def method_two_predict(self, i, X):
        """
        use classifiers to predict class probabilities for X
        use second model to predict X
        """
        bFR = True
        predicts = []
        for j in range(len(self.base_classifiers)):
            if j != i:
                predict = self.different_predict_proba(j, X)
                predict = pd.DataFrame(predict)
                if bFR:
                    predicts = predict
                    bFR = False
                else:
                    predicts = pd.concat([predicts, predict], axis = 1)
        y_pred, new_X_Li = self.second_model_predict(i, predicts)
        return y_pred, new_X_Li

    def method_two_measure_error(self, i):
        """
        use classifiers(except the classifier i) to predict class probabilities for labeled data 
        use the probabilities to train the second model
        if use_auto_select_threshold:
            use the second model to predict labeled data with classification threshold which is auto selected
        else:
            use the second model to predict labeled data
        error_rate = the rate of labeled data that are mislabeled 
        """
        indexs = []
        for j in range(len(self.base_classifiers)): 
            if j != i:
                indexs.append(j)
        predicts = self.generate_confidence_table(indexs)
        self.second_model[i] = self.build_second_model(predicts, self.y_labeled)
        self.second_model_fit(i, predicts, self.y_labeled)
        if self.use_auto_select_threshold:
            probas = self.second_model[i].predict_proba(predicts)
            self.threshold_dict = self.auto_select.auto_select_threshold_fit(
                                         self.second_model[i], probas, self.y_labeled)
            probas = self.second_model[i].predict_proba(predicts)
            y_pred = self.auto_select.auto_select_threshold_predict(
                                        self.second_model[i], probas, y_test = self.y_labeled,
                                        threshold_dict = self.threshold_dict)
        else:
            y_pred_temp = pd.DataFrame(self.second_model[i].predict(predicts))
            y_pred = []
            class_index_dict = {}
            for ele in range(len(self.y_labeled.cat.categories)):
                class_index_dict.update({ele:self.y_labeled.cat.categories[ele]})
            for ele in range(len(y_pred_temp)):
                y_pred.append(class_index_dict[np.argmax(y_pred_temp.loc[ele, :])])
        accuracy = accuracy_score(list(self.y_labeled), y_pred)
        error_rate = 1 - accuracy
        return error_rate
    
    def different_fit(self, i, X, y):
        if self.feature_pre_dict[i] != None:
            if self.feature_pre_dict[i] == "PCA":
                self.feature_pre_models[i] = eval(self.setting_dict[self.feature_pre_dict[i]])
                self.feature_pre_models[i].fit(X)
                X = self.feature_pre_models[i].transform(X)
            else:
                print("feature pre error!")

        if self.base_classifiers_str[i] in ["NB", "AdaBoost", "XgBoost", "SVM", "RF", "KNN", "DT"]:
            model = eval(self.setting_dict[self.base_classifiers_str[i]])
            model.fit(X, y)
            return model
        else:
            pass
    
    def different_predict(self, i, X):
        if self.feature_pre_dict[i] != None:
            if self.feature_pre_dict[i] == "PCA":
                X = self.feature_pre_models[i].transform(X)
            else:
                print("feature_pre error!")
     
        if self.base_classifiers_str[i] in ["NB", "AdaBoost", "XgBoost", "SVM", "RF", "KNN", "DT"]:
            predict = self.base_classifiers[i].predict(X)
            predict = predict.tolist()
            return predict
        else:
            pass

    def different_predict_proba(self, i, X):
        if self.feature_pre_dict[i] != None:
            if self.feature_pre_dict[i] == "PCA":
                X = self.feature_pre_models[i].transform(X)
            else:
                print("feature pre error")

        if self.base_classifiers_str[i] in ["NB", "AdaBoost", "XgBoost", "SVM", "RF", "KNN", "DT"]:
            predict = self.base_classifiers[i].predict_proba(X)
            predict = predict.tolist()
            return predict
        else:
            pass
        
    def remove_Li(self, X, y, remove_num):
        X = np.array(X)
        y = np.array(y)
        remove_list = np.random.choice(len(X), replace = False, size = min(len(X), remove_num))
        X = np.delete(X, remove_list, axis = 0)
        y = np.delete(y, remove_list, axis = 0)
        return X, y
        
    def subsample(self, pre_error_rate, pre_Li_ln, error_rate, X_Li, y_Li):
        s = math.ceil(pre_error_rate * pre_Li_ln / error_rate - 1)
        X_Li, y_Li = self.remove_Li(X_Li, y_Li, len(X_Li) - s)
        return X_Li, y_Li

    def generate_confidence_table(self, indexs):
        predicts = []
        bFR = True
        for index in indexs:
            predict = self.different_predict_proba(index, self.X_labeled)
            predict = pd.DataFrame(predict)
            if bFR:
                predicts = predict
                bFR = False
            else:
                predicts = pd.concat([predicts, predict], axis = 1)
        return predicts
     
    def retrain(self, i, X_Li, y_Li):
        model = eval(self.setting_dict[self.base_classifiers_str[i]])
        X_Li = pd.DataFrame(X_Li, columns = self.X_labeled.columns)
        X_combine = self.X_labeled.append(X_Li)
        self.X_combine[i] = X_combine
        y_Li = pd.Series(y_Li, dtype = "category")
        y_Li.cat.set_categories(self.y_labeled.cat.categories, inplace = True)
        y_combine = self.y_labeled.append(y_Li)
        self.y_combine[i] = y_combine
        model = self.different_fit(i, X_combine, y_combine)
        return model
    
    def train(self):
        bFR = True
        #initialize error rate, previous error rate, and previous length of Li 
        pre_error_rate = [0.5 for ele in range(len(self.base_classifiers))]
        error_rate = [0.0 for ele in range(len(self.base_classifiers))]
        pre_Li_ln = [0.0 for ele in range(len(self.base_classifiers))]
        turn_start_time = time.time()
        turn = 0
        while True:
            X_Li = [[] for ele in range(len(self.base_classifiers))]
            y_Li = [[] for ele in range(len(self.base_classifiers))]
            update = [False for ele in range(len(self.base_classifiers))]
            for i in range(len(self.base_classifiers)):     
                #use second model to measure error       
                error_rate[i] = self.method_two_measure_error(i)
                if error_rate[i] < pre_error_rate[i]:
                    #use second model to predict unlabeled data
                    predicts, new_X_Li = self.method_two_predict(i, self.X_unlabeled)
                    X_Li[i] = new_X_Li
                    y_Li[i] = predicts
                        
                    if pre_Li_ln[i] == 0:
                        pre_Li_ln[i] = math.floor(error_rate[i] / (pre_error_rate[i] - error_rate[i]) + 1)
                    """
                    if error rate * length of Li < previous error rate * length of Li:
                        use labeled data and unlabeled data Li to retrain classifiers
                    elif length of Li is too big:
                        use subsampling to reduce the size of Li
                        use labeled data and unlabeled data Li to retrain classifiers
                    """
                    if pre_Li_ln[i] < len(X_Li[i]):
                        if error_rate[i] * len(X_Li[i]) < pre_error_rate[i] * pre_Li_ln[i]:
                            update[i] = True
                            self.turn_nums[i] += 1
                            self.unlabeled_nums[i] += len(X_Li[i])
                        elif pre_Li_ln[i] > error_rate[i] / (pre_error_rate[i] - error_rate[i]):                            
                            X_Li[i], y_Li[i] = self.subsample(pre_error_rate[i], pre_Li_ln[i], error_rate[i], X_Li[i], y_Li[i])
                            update[i] = True
                            self.turn_nums[i] += 1
                            self.unlabeled_nums[i] += len(X_Li[i])
                        else:
                            pass
            turn += 1
            # if all classifiers don't update this turn: stop the algorithm 
            if True not in update:
                turn_end_time = time.time()
                total_time = str(round((turn_end_time - turn_start_time) / 60,1))
                for i in range(len(self.unlabeled_nums)):
                    if self.unlabeled_nums[i] != 0:
                        self.unlabeled_num_sum += self.unlabeled_nums[i] / self.turn_nums[i]
                break
            for i in range(len(self.base_classifiers)):
                if update[i]:
                    self.base_classifiers[i] = self.retrain(i, X_Li[i], y_Li[i]) 
                pre_error_rate[i] = error_rate[i]
                pre_Li_ln[i] = len(X_Li)
    
    def select_max_category(self, vote_dict, proba_dict):
        max_vote = max(vote_dict.items(), key = lambda ele : ele[1])[1]
        max_categories = []
        for key, value in vote_dict.items():
            if value == max_vote:
                max_categories.append(key)
        if len(max_categories) > 1:
            compare_list = []
            for category in max_categories:
                compare_list.append([category, proba_dict[category]])
            return max(compare_list, key = lambda ele : ele[1])[0]
        else:
            return max_categories[0]

    def majority_vote(self, X_test):
        predicts = []
        predicts_proba = []
        y_pred = []
        for i in range(len(self.base_classifiers)):
            predicts.append(self.different_predict(i, X_test))
            predicts_proba.append(self.different_predict_proba(i, X_test))
            
        for k in range(len(predicts[0])):
            count_dict = {}
            proba_dict = {}
            for i in range(len(predicts)):
                temp = count_dict.get(predicts[i][k])
                if temp != None:
                    count_dict[predicts[i][k]] += 1
                else:
                    count_dict.update({predicts[i][k]:1})
                temp = proba_dict.get(predicts[i][k])
                if temp != None:
                    proba_dict[predicts[i][k]] += predicts_proba[i][k]
                else:
                    proba_dict.update({predicts[i][k] : predicts_proba[i][k]})
            y_pred.append(self.select_max_category(count_dict, proba_dict))
        return y_pred
  
    def evaluate(self, X_test, y_test, avg_type = "weighted", save_pred = True):
        y_pred = self.majority_vote(X_test)
        y_test = y_test.tolist()
        results = []
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average = avg_type)
        score_dict = {}
        score_dict.update({"precision":precision})
        score_dict.update({"recall":recall})
        score_dict.update({"fscore":fscore})
        score_dict.update({"support":support})
        return score_dict, y_pred, self.unlabeled_num_sum
    

if __name__ == "__main__":
    #algorithm parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_auto_select_threshold",
                        action = "store_true",
                        help = "whether to use auto select threshold when predict unlabeled data")
    parser.add_argument("-g", "--groups", type = str, nargs = "+",
                        default = ["NB_AdaBoost_DT"],                        
                        help = """
                        the type of base classifiers can be different
                        use _ to separate the classifier type
                        """)
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
    #data and log parameters
    parser.add_argument("--feature_pre", type = str,
                        choices = [None, "PCA"],
                        default = None,
                        help = "use principal component analysis for transforming the features")
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
                        default = os.path.join(os.getcwd(), "log", "two_phase_auto_fit={}_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())),
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
    args.log_dir = args.log_dir.format(args.use_auto_select_threshold)
    os.mkdir(args.log_dir)
    for i in range(len(args.groups)):
        args.groups[i] = args.groups[i].split("_")
    data_pre = DataPreprocessing()
    total_start_time = time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime())
    if args.data_dir == None:
        dataset_start = datetime.datetime.now()
        if args.save_pred:
            log_dict = {
            "dataset":[],"group":[],"confidence_threshold":[],"label_rate":[],"avg_type":[],"precision":[],"recall":[],"fscore":[],"cv":[],
            "y_pred":[],"y_test":[],"unlabeled_num":[]
            }
        else:
            log_dict = {
            "dataset":[],"group":[],"confidence_threshold":[],"label_rate":[],"avg_type":[],"precision":[],"recall":[],"fscore":[],"cv":[]
            }
        dataset = load_breast_cancer()
        X = dataset["data"]
        y = dataset["target"]
        X = pd.DataFrame(X)
        y = pd.Series(y)
        if args.data_pre_type == "category_and_numeric":
            X = data_pre.category_and_numeric(X)
        elif args.data_pre_type == "all_category":
            X = data_pre.columns_category(X)
        else:
            print("data_pre_type error!")
        X = pd.get_dummies(X)
        y = data_pre.to_category(y)
        bEnd = False
        for ele in range(args.experiment_num):
            print("experiment_num start : " + str(ele))
            experiment_num_start = datetime.datetime.now()
            for label_rate in args.label_rates:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = args.train_size, stratify = y, random_state = ele) 
                X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size = label_rate, stratify = y_train, random_state = ele)
                for group in args.groups:
                    feature_pre_dict = {}
                    for i in range(len(group)):
                        feature_pre_dict.update({i : args.feature_pre})
                    print("group start:" + "_".join(group))
                    group_start = datetime.datetime.now()
                    for confidence_threshold in args.confidence_thresholds:      
                        two_phase_auto_fit = TwoPhaseAutoFit(group, X_labeled, y_labeled, X_unlabeled, args.data_pre_type, 
                        use_auto_select_threshold = args.use_auto_select_threshold,
                        confidence_threshold = confidence_threshold
                        )
                        print("train start : " + time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime()))
                        train_start = datetime.datetime.now()
                        two_phase_auto_fit.train()
                        train_end = datetime.datetime.now()
                        print("seconds : " + str((train_end - train_start).seconds))
                        print("train end : " + time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime()))
                        for avg_type in args.avg_types:
                            score_dict, y_pred, unlabeled_num = two_phase_auto_fit.evaluate(X_test, y_test, avg_type)
                            log_dict["dataset"].append("breast_cancer_w")
                            log_dict["group"].append("_".join(group))
                            log_dict["confidence_threshold"].append(confidence_threshold)
                            log_dict["label_rate"].append(label_rate)
                            log_dict["avg_type"].append(avg_type)
                            log_dict["precision"].append(score_dict["precision"])
                            log_dict["recall"].append(score_dict["recall"])
                            log_dict["fscore"].append(score_dict["fscore"])
                            log_dict["cv"].append(-1)
                            if args.save_pred:
                                log_dict["y_pred"].append(y_pred)
                                log_dict["y_test"].append(y_test.tolist())
                                log_dict["unlabeled_num"].append(unlabeled_num)
                    group_end = datetime.datetime.now()
                    print("seconds : " + str((group_end - group_start).seconds))
                    print("group end : " + "_".join(group))
            experiment_num_end = datetime.datetime.now()
            print("seconds : " + str((experiment_num_end - experiment_num_start).seconds))
            print("experiment_num end")
            if bEnd:
                break
        if not bEnd:
            data_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + "breast_cancer_w.csv"
            df_log = pd.DataFrame(log_dict)
            df_log.to_csv(os.path.join(args.log_dir, data_name), index = False)
    else:    
        for data in glob.glob(os.path.join(args.data_dir, "*.csv")):
            dataset_start = datetime.datetime.now()
            print("dataset: " + os.path.basename(data))
            if args.save_pred:
                log_dict = {
                "dataset":[],"group":[],"confidence_threshold":[],"label_rate":[],"avg_type":[],"precision":[],"recall":[],"fscore":[],"cv":[],
                "y_pred":[],"y_test":[],"unlabeled_num":[]
                }
            else:
                log_dict = {
                "dataset":[],"group":[],"confidence_threshold":[],"label_rate":[],"avg_type":[],"precision":[],"recall":[],"fscore":[],"cv":[]
                }
            df = pd.read_csv(data)
            if args.data_pre_type == "category_and_numeric":
                df = data_pre.category_and_numeric(df)
            elif args.data_pre_type == "all_category":
                df = data_pre.columns_category(df)
            else:
                print("data_pre_type error!")
            X = df[df.columns[:-1]]
            X = pd.get_dummies(X)
            y = df[df.columns[-1]]
            bEnd = False
            for ele in range(args.experiment_num):
                print("experiment_num start : " + str(ele))
                experiment_num_start = datetime.datetime.now()
                for label_rate in args.label_rates:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = args.train_size, stratify = y, random_state = ele) 
                    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size = label_rate, stratify = y_train, random_state = ele)
                    for group in args.groups:
                        feature_pre_dict = {}
                        for i in range(len(group)):
                            feature_pre_dict.update({i : args.feature_pre})
                        print("group start:" + "_".join(group))
                        group_start = datetime.datetime.now()
                        for confidence_threshold in args.confidence_thresholds:      
                            two_phase_auto_fit = TwoPhaseAutoFit(group, X_labeled, y_labeled, X_unlabeled, args.data_pre_type, 
                            use_auto_select_threshold = args.use_auto_select_threshold,
                            confidence_threshold = confidence_threshold
                            )
                            print("train start : " + time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime()))
                            train_start = datetime.datetime.now()
                            two_phase_auto_fit.train()
                            train_end = datetime.datetime.now()
                            print("seconds : " + str((train_end - train_start).seconds))
                            print("train end : " + time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime()))
                            for avg_type in args.avg_types:
                                score_dict, y_pred, unlabeled_num = two_phase_auto_fit.evaluate(X_test, y_test, avg_type)
                                log_dict["dataset"].append(os.path.basename(data))
                                log_dict["group"].append("_".join(group))
                                log_dict["confidence_threshold"].append(confidence_threshold)
                                log_dict["label_rate"].append(label_rate)
                                log_dict["avg_type"].append(avg_type)
                                log_dict["precision"].append(score_dict["precision"])
                                log_dict["recall"].append(score_dict["recall"])
                                log_dict["fscore"].append(score_dict["fscore"])
                                log_dict["cv"].append(-1)
                                if args.save_pred:
                                    log_dict["y_pred"].append(y_pred)
                                    log_dict["y_test"].append(y_test.tolist())
                                    log_dict["unlabeled_num"].append(unlabeled_num)
                        group_end = datetime.datetime.now()
                        print("seconds : " + str((group_end - group_start).seconds))
                        print("group end : " + "_".join(group))
                experiment_num_end = datetime.datetime.now()
                print("seconds : " + str((experiment_num_end - experiment_num_start).seconds))
                print("experiment_num end")
                if bEnd:
                    break
            if not bEnd:
                data_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + os.path.basename(data)
                df_log = pd.DataFrame(log_dict)
                df_log.to_csv(os.path.join(args.log_dir, data_name), index = False)
        dataset_end = datetime.datetime.now()
        print("seconds: " + str((dataset_end - dataset_start).seconds))
        print("dataset end")
total_end_time = time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime())
print("finish!")
print(total_start_time)
print(total_end_time)




