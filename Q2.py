import sklearn
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets
# from sklearn.externals import joblibr
import os.path
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.externals.six import StringIO  
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from IPython.display import Image  
from sklearn import tree
import pydotplus
import math


def load_data():
    data_folder = Path("data/Q2/")
    fake_path = data_folder / "clean_fake.txt"
    real_path = data_folder / "clean_real.txt"

    fake_ = open(fake_path,"r")
    fake = fake_.read().split("\n")

    real_ = open(real_path,"r")
    real = real_.read().split("\n")

    size_fake = len(fake)
    size_real = len(real)
    
    # we set fake to 0
    # real to 1 
    fake_target = size_fake * [0]
    real_target = size_real * [1]
    all_news = fake + real
    print("all news size {:d}".format(len(all_news)))
    all_targets = fake_target + real_target
    # print(all_targets)
    vectorizer_all = CountVectorizer()
    all_process = vectorizer_all.fit_transform(all_news)
    # print(all_process)
    toarray_all_news = all_process.toarray()
    print("total X array size: {:d} X {:d}".format(len(toarray_all_news), len(toarray_all_news[0])))
    print("feature name length: {:d}".format(len(vectorizer_all.get_feature_names())))
    #70% training, 15% test, 15% validation
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(toarray_all_news, all_targets, test_size=0.3, random_state=1)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=1)
    print("train, valid, test \n","data size:", len(X_train), len(X_valid), len(X_test), "\ntarget size:", len(y_train), len(y_valid), len(y_test))
    return [[X_train, X_valid, X_test],[y_train, y_valid, y_test]], vectorizer_all.get_feature_names()

def view_data_train(entry_num, data):
    print("data set", data[0][0][entry_num], "target", data[1][0][entry_num])


def validation_test(predict, target):
    total = len(predict)
    correct = 0
    for i in range(0, total, 1):
        if predict[i] == target[i]:
            correct += 1
    return correct/total

def visualize_tree2(model, all_features, depth):
    
    # Create DOT data
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=all_features,  
        filled=True, rounded=True, special_characters=True)

    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)  

    # Show graph
    Image(graph.create_png())

    # Create PNG
    graph.write_png("decision_tree depth "+ str((depth+1)*2)+".png")
    return


def select_model(data, lambda_depth):
    # set IG // Gini
    # max_depth_list = [1, 4, 8, 16, 32]
    # max_depth_list = [1, 4]
    max_depth_list = lambda_depth
    all_models = [[],[]]
    all_preds = [[],[]]
    all_acc = [[],[]]
    count = 0
    for max_depth_i in max_depth_list:
        count += 1
        print("Training data set {:d}  >>>>>>>>>>>>>>>> \n".format(count))
        model_IG = DecisionTreeClassifier(criterion="entropy", max_depth= max_depth_i)
        model_Gini = DecisionTreeClassifier(criterion="gini", max_depth= max_depth_i)
        model_IG = model_IG.fit(data[0][0], data[1][0])
        model_Gini = model_Gini.fit(data[0][0], data[1][0])
        y_IG = model_IG.predict(data[0][1])
        y_Gini = model_Gini.predict(data[0][1])
        acc_IG = validation_test(y_IG, data[1][1])
        acc_Gini = validation_test(y_Gini, data[1][1])
        all_models[0].append(model_IG)
        all_models[1].append(model_Gini)
        all_preds[0].append(y_IG)
        all_preds[1].append(y_Gini)
        all_acc[0].append(acc_IG)
        all_acc[1].append(acc_Gini)
        print("model IG with max_depth {:d}, accuracy is {:f} \n".format(max_depth_i, acc_IG))
        print("model Gini with max_depth {:d}, accuracy is {:f} \n".format(max_depth_i, acc_Gini))
    max_IG = max(all_acc[0])
    max_Gini = max(all_acc[1])

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("model IG is max with index {:d}, accuracy of {:f} \n".format(all_acc[0].index(max_IG), max_IG))
    print("model Gini is max with index {:d}, accuracy of {:f} \n".format(all_acc[1].index(max_Gini), max_Gini))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    is_IG = 0 if (max_IG>max_Gini) else 1
    index = all_acc[0].index(max_IG) if(max_IG > max_Gini) else all_acc[1].index(max_Gini)
    return all_acc, all_models, is_IG, index

def compute_infomation_gain(Y, xi, training_x, feature):
    # feature is all features in all models
    # where Y is the result, xi is a feature
    fake_count = 0
    real_count = 0
    found = False
    for i in range(len(feature)):
        if feature[i] == xi:
            print("found feature {:s}, at position index {:d}".format(xi, i))
            feature_index = i
            found = True
    # array X convention: a X b, X[a][b], where a is index of news, b is index of features
    # y length is same as a
    if(found == False):
        print("feature does not exist")
        return -1
    for i in Y:
        if (i == 1):
            real_count += 1
        else:
            fake_count += 1
    fake_feature_count = 0
    fake_not_feature_count = 0
    real_feature_count = 0
    real_not_feature_count = 0
    for i in range(len(Y)):
        if(training_x[i][feature_index] >= 1):
            # is feature
            if(Y[i] == 0):
                fake_feature_count += 1
            else:
                real_feature_count += 1
        else:
            # is not feature
            if(Y[i] == 0):
                fake_not_feature_count += 1
            else:
                real_not_feature_count += 1
    Root_Entropy = (-1)*real_count/(real_count+fake_count)*math.log(real_count/(real_count+fake_count), 2)+(-1)*fake_count/(real_count+fake_count)*math.log(fake_count/(real_count+fake_count), 2)
    feature_pro = (fake_feature_count+real_feature_count)/(fake_count+real_count)
    non_feature_pro = (fake_not_feature_count+real_not_feature_count)/(fake_count+real_count)
    if(real_feature_count == 0):
        real_feature_entropy = 0
    else:
        real_feature_entropy = (-1)*real_feature_count/(real_feature_count+fake_feature_count)*math.log(real_feature_count/(real_feature_count+fake_feature_count), 2)
    if(fake_feature_count == 0):
        fake_feature_entropy = 0
    else:
        fake_feature_entropy = (-1)*fake_feature_count/(real_feature_count+fake_feature_count)*math.log(fake_feature_count/(real_feature_count+fake_feature_count), 2)
    if(real_not_feature_count == 0):
        real_no_feature_entropy = 0
    else:
        real_no_feature_entropy = (-1)*real_not_feature_count/(real_not_feature_count+fake_not_feature_count)*math.log(real_not_feature_count/(real_not_feature_count+fake_not_feature_count), 2)
    if(fake_not_feature_count == 0):
        fake_no_feature_entropy = 0
    else:
        fake_no_feature_entropy = (-1)*fake_not_feature_count/(real_not_feature_count+fake_not_feature_count)*math.log(fake_not_feature_count/(real_not_feature_count+fake_not_feature_count), 2)
    
    feature_entropy = real_feature_entropy + fake_feature_entropy
    non_feature_entropy = real_no_feature_entropy + fake_no_feature_entropy
    Leaf_Entropy = feature_pro * feature_entropy  + non_feature_pro * non_feature_entropy
    IG = Root_Entropy - Leaf_Entropy
    return IG

if __name__ == "__main__":
    data, all_features = load_data ()
    #data: [[X_train, X_valid, X_test],[y_train, y_valid, y_test]]
    # tryout_count = 1
    # tryout_list = [0]*tryout_count
    # for i in range(0, tryout_count, 1):
    #     tryout_list[i] = 2**i
    tryout_list = [2]

    print("try out list for max_depth: ", tryout_list)
    all_acc, all_models, is_IG, index = select_model(data, tryout_list)
    
    ## model IG with depth 64 fits the best, visualize it

    visualize_tree2(all_models[is_IG][index], all_features, index)
    features_to_search = ["find", "trump", "donald", "and", "hillary",'clinton',"at","rick", "and", "morty","season","eight"]
    for feature_to_search in features_to_search:
        IG = compute_infomation_gain(data[1][0], feature_to_search, data[0][0], all_features)
        if(IG == -1):
            print(">>IG for feature [{:s}] is not found".format(feature_to_search))
        else:
            print(">>IG for feature [{:s}] is {:f}".format(feature_to_search, IG))
