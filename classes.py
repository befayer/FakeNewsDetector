# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import nltk
import pandas as pd
from boto import sns
from nltk import tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sqlite3
import datetime
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets


class BD:

    def loaddata(self, tableWidge):
    connection = sqlite3.connect('resourse/history.db')
    cursor = connection.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS news ("title" TEXT,"text" TEXT,"result" TEXT, "data" TEXT)""")
    sqlShowData = "SELECT * FROM news LIMIT 50"

    tableWidget.setRowCount(50)
    tableRow = 0
    for row in cursor.execute(sqlShowData):
        self.tableWidget.setItem(tableRow, 0, QtWidgets.QTableWidgetItem(row[0]))
        self.tableWidget.setItem(tableRow, 1, QtWidgets.QTableWidgetItem(row[1]))
        self.tableWidget.setItem(tableRow, 2, QtWidgets.QTableWidgetItem(row[2]))
        self.tableWidget.setItem(tableRow, 3, QtWidgets.QTableWidgetItem(row[3]))
        tableRow += 1

    cursor.close()


def add_functions(self):
    self.pushButton.clicked.connect(
        lambda: self.result(self.textTitleNews.toPlainText(), self.textNews.toPlainText()))


def add_functions_test(self):
    self.pushButton_test.clicked.connect(lambda: self.result_test())


# def btn_clear_history(self):
# self.pushButton_clearHistoryDB.clicked.connect(lambda: self.clearHistory())

def clearHistory(self):
    connection = sqlite3.connect('resourse/history.db')
    cursor = connection.cursor()
    sqlClearData = "DELETE FROM news"
    cursor.execute(sqlClearData)
    connection.commit()
    connection.close()
    cursor.close()
    self.loaddata()

def result_test(self):
    # getting values

    form_test_size = round(self.doubleSpinBoxSizeTestSet.value(), 1)
    form_max_df = round(self.doubleSpinBoxMaxDF.value(), 1)
    form_max_iter = self.spinBox.value()

    print(form_test_size)
    print(form_max_df)
    print(form_max_iter)

    # Read the data
    df_test = pd.read_csv('resourse\\news.csv')
    # Get shape and head
    df_test.shape
    df_test.head()

    # DataFlair - Get the labels
    labels_test = df_test.label
    labels_test.head()

    # DataFlair - Split the dataset
    # x_train, x_test, y_train, y_test = train_test_split(df_test['text'], labels_test, test_size=0.2, random_state=7)
    x_train, x_test, y_train, y_test = train_test_split(df_test['text'], labels_test, test_size=form_test_size,
                                                        random_state=7)

    # DataFlair - Initialize a TfidfVectorizer
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=form_max_df)

    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # DataFlair - Initialize a PassiveAggressiveClassifier
    # pac = PassiveAggressiveClassifier(max_iter=50)
    pac = PassiveAggressiveClassifier(max_iter=form_max_iter)
    pac.fit(tfidf_train, y_train)

    # DataFlair - Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')

    # DataFlair - Build confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    result_test = list(matrix)
    temp1 = result_test[0]
    temp2 = result_test[1]
    temp11 = list(temp1)
    temp22 = list(temp2)

    temp111 = temp11[0]  # 1
    temp112 = temp11[1]  # 2
    temp121 = temp22[0]  # 3
    temp122 = temp22[1]  # 4

    # print(temp111)
    # print(temp112)
    # print(temp121)
    # print(temp122)
    # print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

    # Out
    temp_label_accuracy = f'Accuracy: {round(score * 100, 2)}%'
    self.label_accuracy.setText(temp_label_accuracy)
    self.label_predskazano_pravilno.setText('?????????????????????? ??????????????????:')

    self.label_pravilno_count_true_result.setText(f'{temp111} true-??????????????????????')
    self.label_pravilno_count_fake_result.setText(f'{temp122} fake-??????????????????????')
    self.label_predskazano_neverno.setText('?????????????????????? ??????????????:')
    self.label_neverno_count_true_result.setText(f'{temp112} true-??????????????????????')
    self.label_neverno_count_fake_result.setText(f'{temp121} fake-??????????????????????')
    count_neverno = temp112 + temp121
    self.label_9.setText(
        f'?????????????? ????????????????????, ?????? {temp111} ???????????????? true, {temp122} ???????????????? fake. ???? ?????? {count_neverno} ???????????????? ???????? ???????????????? ???? ?? ???????? ????????????.')

    # self.drawGraphic(temp111,temp122,temp112,temp121)
    self.drawGraphicTest(temp111, temp122, temp112, temp121)

    X = tfidf_vectorizer.fit_transform(df_test['text'])
    # zipping actual words and sum of their Tfidf for corpus
    features_rank = list(zip(tfidf_vectorizer.get_feature_names(), [x[0] for x in X.sum(axis=0).T.tolist()]))

    # sorting
    features_rank = np.array(sorted(features_rank, key=lambda x: x[1], reverse=True))

    n = 10
    plt.figure(figsize=(5, 10))
    plt.barh(-np.arange(n), features_rank[:n, 1].astype(float), height=.8)
    plt.yticks(ticks=-np.arange(n), labels=features_rank[:n, 0])
    plt.savefig('graphicsubjest.png', dpi=100)
    # self.label_2.setPixmap(QtGui.QPixmap("graphicsubjest.png"))


def drawGraphicTest(self, predictedTrueTrue, predictedTrueFake, predictedFalseTrue, predictedFalseFake):
    cat_par = ["???????????????????????????????? ??????????"]
    # g1 = [predictedTrueTrue, predictedTrueFake]
    # g2 = [predictedFalseTrue, predictedFalseFake]

    g1 = [predictedTrueTrue]
    g2 = [predictedTrueFake]

    width = 0.2
    x = np.arange(len(cat_par))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, g1, width, color='green', label='TRUE-????????????????')
    rects2 = ax.bar(x + width / 2, g2, width, color='orange', label='FALSE-????????????????')
    ax.set_title('?????????????????? ???????????????????????????????????? ????????????????')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_par)
    ax.legend()
    fig.savefig('graphic.png')
    self.label_3.setPixmap(QtGui.QPixmap("graphic.png"))


def result(self, title, text):
    df = pd.read_csv('resourse\\news.csv')
    columns = ['title', 'text', 'label']
    data = [[title, text, '']]
    dff = pd.DataFrame(data, columns=columns)
    dff.to_csv('D:\\in.csv')
    dfin = pd.read_csv('D:\\in.csv')
    df.shape
    dfin.shape
    labels = df.label
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2,
                                                        random_state=7)  # random_state - ?????????????????? ????????????????????????????
    x_test_in = dfin['text']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                       max_df=0.7)  # max_df = 0.7 ???????????????? " ???????????????????????? ??????????????, ?????????????? ?????????????????????? ?????????? ?????? ?? 70% ???????????????????? ".
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test_in)
    pac = PassiveAggressiveClassifier(max_iter=50)  # ???????????????????????? ???????????????????? ???????????????? ???? ?????????????????? ????????????
    pac.fit(tfidf_train, y_train)
    y_pred = pac.predict(tfidf_test)
    print("??????????????????:")
    print(y_pred)
    result = list(y_pred)
    for i in range(len(result)):
        if (result[i] == "FAKE"):
            self.label_result.setStyleSheet("font: 75 35pt \"Berlin Sans FB Demi\"; color: rgb(229, 0, 0);;")
        else:
            self.label_result.setStyleSheet("font: 75 35pt \"Berlin Sans FB Demi\"; color: rgb(20, 207, 0);")
        self.label_result.setText(result[i])

    connection = sqlite3.connect('resourse/history.db')
    cursor = connection.cursor()
    now = datetime.datetime.now()
    currentDate = now.strftime("%d-%m-%Y %H:%M")
    short_text = text[0:120]
    short_title = title[0:120]
    print(short_text)
    print(short_title)
    print(currentDate)
    cursor.execute("""CREATE TABLE IF NOT EXISTS news ("title" TEXT,"text" TEXT, "result" TEXT, "data" TEXT)""")
    sqlAddData = f"INSERT INTO news VALUES ('{short_title}', '{short_text}','{result[i]}','{currentDate}')"
    cursor.execute(sqlAddData)
    connection.commit()
    cursor.close()
    self.loaddata()