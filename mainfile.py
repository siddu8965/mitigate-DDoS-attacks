import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import webbrowser

st.title("Mitigating DDOS Attack In IOT Network Environment")

@st.cache
def getLabel(name):
    label = -1
    for i in range(len(labels)):
        if name == labels[i]:
            label = i
            break
    return label

def uploadDataset():
    uploaded_file = st.file_uploader("Upload DDOS Dataset", type="csv")
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        st.write(dataset.head())
        return dataset

def preprocessDataset(dataset):
    global label_encoder, X, Y, pca, X_train, X_test, y_train, y_test
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object' and columns[i] != 'Label':
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            label_encoder.append(le)
    dataset.fillna(0, inplace=True)
    Y = dataset['Label'].ravel()
    temp = []
    for i in range(len(Y)):
        temp.append(getLabel(Y[i]))
    temp = np.asarray(temp)
    Y = temp
    dataset = dataset.values
    X = dataset[:, 0:dataset.shape[1]-1]
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    st.write(f"{algorithm} Accuracy  :  {a}")
    st.write(f"{algorithm} Precision : {p}")
    st.write(f"{algorithm} Recall    : {r}")
    st.write(f"{algorithm} FScore    : {f}")
    st.write("\n")
    st.write(np.unique(predict))
    st.write(np.unique(y_test))
    conf_matrix = confusion_matrix(y_test, predict)
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    st.pyplot(plt)

def runNaiveBayes():
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predict = nb.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)

def runRandomForest():
    if os.path.exists('model/rf.txt'):
        with open('model/rf.txt', 'rb') as file:
            rf = pickle.load(file)
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        with open('model/rf.txt', 'wb') as file:
            pickle.dump(rf, file)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

def runSVM():
    if os.path.exists('model/svm.txt'):
        with open('model/svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
    else:
        svm_cls = svm.SVC()
        svm_cls.fit(X_train, y_train)
        with open('model/svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def runXGBoost():
    if os.path.exists('model/xgb.txt'):
        with open('model/xgb.txt', 'rb') as file:
            xgb_cls = pickle.load(file)
    else:
        xgb_cls = XGBClassifier()
        xgb_cls.fit(X_train, y_train)
        with open('model/xgb.txt', 'wb') as file:
            pickle.dump(xgb_cls, file)
    predict = xgb_cls.predict(X_test)
    calculateMetrics("XGBoost", predict, y_test)

def runAdaBoost():
    if os.path.exists('model/adb.txt'):
        with open('model/adb.txt', 'rb') as file:
            adb_cls = pickle.load(file)
    else:
        adb_cls = AdaBoostClassifier()
        adb_cls.fit(X_train, y_train)
        with open('model/adb.txt', 'wb') as file:
            pickle.dump(adb_cls, file)
    predict = adb_cls.predict(X_test)
    calculateMetrics("AdaBoost", predict, y_test)

def runKNN():
    if os.path.exists('model/knn.txt'):
        with open('model/knn.txt', 'rb') as file:
            knn_cls = pickle.load(file)
    else:
        knn_cls = KNeighborsClassifier(n_neighbors=2)
        knn_cls.fit(X_train, y_train)
        with open('model/knn.txt', 'wb') as file:
            pickle.dump(knn_cls, file)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN", predict, y_test)

def predict():
    filename = st.file_uploader("Upload Test Data", type="csv")
    if filename is not None:
        testData = pd.read_csv(filename)
        count = 0
        for i in range(len(types)-1):
            name = types[i]
            if name == 'object':
                if columns[i] == 'Flow Bytes/s':
                    testData[columns[i]] = pd.Series(label_encoder[count].fit_transform(testData[columns[i]].astype(str)))
                else:
                    testData[columns[i]] = pd.Series(label_encoder[count].transform(testData[columns[i]].astype(str)))
                count = count + 1
        testData.fillna(0, inplace=True)
        testData = testData.values
        testData = normalize(testData)
        testData = pca.transform(testData)
        predict = classifier.predict(testData)
        for i in range(len(predict)):
            st.write(f"Test DATA : {testData[i]} ===> PREDICTED AS {labels[predict
