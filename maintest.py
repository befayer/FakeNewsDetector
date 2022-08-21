# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import itertools
import re

import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from PyQt5 import QtCore, QtGui, QtWidgets


class MachineLearning:

    def __init__(self, label_accuracy, label_predskazano_pravilno, label_pravilno_count_true_result, label_pravilno_count_fake_result,
                 label_predskazano_neverno, label_neverno_count_true_result, label_neverno_count_fake_result, label_result):
        self.label_accuracy = label_accuracy
        self.label_predskazano_pravilno = label_predskazano_pravilno
        self.label_pravilno_count_true_result = label_pravilno_count_true_result
        self.label_pravilno_count_fake_result = label_pravilno_count_fake_result
        self.label_predskazano_neverno = label_predskazano_neverno
        self.label_neverno_count_true_result = label_neverno_count_true_result
        self.label_neverno_count_fake_result = label_neverno_count_fake_result
        self.label_result = label_result

        # устанавливаем имя

    def result_test(self):
        # Read the data
        df_test = pd.read_csv('resourse\\news.csv')
        # Get shape and head
        df_test.shape
        df_test.head()

        # DataFlair - Get the labels
        labels_test = df_test.label
        labels_test.head()

        # DataFlair - Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(df_test['text'], labels_test, test_size=0.2, random_state=7)

        # DataFlair - Initialize a TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

        # DataFlair - Fit and transform train set, transform test set
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = tfidf_vectorizer.transform(x_test)

        # DataFlair - Initialize a PassiveAggressiveClassifier
        pac = PassiveAggressiveClassifier(max_iter=50)
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

        temp111 = temp11[0] #1
        temp112 = temp11[1] #2
        temp121 = temp22[0] #3
        temp122 = temp22[1] #4

        print(temp111)
        print(temp112)
        print(temp121)
        print(temp122)
        print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

        # Out
        temp_label_accuracy = f'Accuracy: {round(score * 100, 2)}%'
        self.label_accuracy.setText(temp_label_accuracy)
        self.label_predskazano_pravilno.setText('Предсказано правильно:')

        self.label_pravilno_count_true_result.setText(f'{temp111} true-результатов')
        self.label_pravilno_count_fake_result.setText(f'{temp122} fake-результатов')
        self.label_predskazano_neverno.setText('Предсказано неверно:')
        self.label_neverno_count_true_result.setText(f'{temp112} true-результатов')
        self.label_neverno_count_fake_result.setText(f'{temp121} fake-результатов')

    def result(self, title, text):
        df = pd.read_csv('resourse\\news.csv')
        columns = ['title', 'text', 'label']
        data = [[title, text, '']]

        dff = pd.DataFrame(data, columns=columns)
        dff.to_csv('D:\\in.csv')
        dfin = pd.read_csv('D:\\in.csv')

        # Get the shape
        df.shape
        dfin.shape
        print("------")
        print("df.head")
        print(df.head(10))
        print("------")

        print("------")
        print("df_in.head")
        print(dfin)
        print("------")

        # DataFlair - Get the labels
        labels = df.label
        print(labels.head())

        # DataFlair - Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
        x_test_in = dfin['text']
        print("this is x_test_in")
        # df[text] - DataFrame
        # labels -
        # test_size -
        # random_state=Управляет перемешиванием, применяемым к данным перед применением разделения.)

        # DataFlair - Initialize a TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                               max_df=0.7)  # Матрица фукнций (стоп-слово, 0.7 - автоматическое обнаружение стоп-слова)
        # DataFlair - Fit and transform train set, transform test set
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)  # центрирование обучающего набора
        tfidf_test = tfidf_vectorizer.transform(x_test_in)  # перемешивание данных

        # DataFlair - Initialize a PassiveAggressiveClassifier
        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train, y_train)
        # DataFlair - Predict on the test set and calculate accuracy
        y_pred = pac.predict(tfidf_test)
        # прогнозирование

        print("полученные лейблы:")
        print(y_pred)
        result = list(y_pred)
        for i in range(len(result)):
            self.label_result.setText(result[i])
            # self.label_3.setText(result[i])


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(558, 622)
        MainWindow.setStyleSheet("background-color: rgb(229, 229, 229);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 561, 641))
        self.tabWidget.setStyleSheet("background-color: rgb(167, 167, 167);\n"
"border-color: rgb(0, 0, 0);")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(60, 30, 461, 31))
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("\n"
"font: 20pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label.setObjectName("label")
        self.label_title = QtWidgets.QLabel(self.tab_2)
        self.label_title.setGeometry(QtCore.QRect(240, 90, 81, 16))
        self.label_title.setStyleSheet("\n"
"font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_title.setObjectName("label_title")
        self.textTitleNews = QtWidgets.QTextEdit(self.tab_2)
        self.textTitleNews.setGeometry(QtCore.QRect(30, 110, 511, 61))
        self.textTitleNews.setStyleSheet("background-color: rgb(229, 229, 229);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Berlin Sans FB Demi\";")
        self.textTitleNews.setObjectName("textTitleNews")
        self.label_text = QtWidgets.QLabel(self.tab_2)
        self.label_text.setGeometry(QtCore.QRect(260, 190, 41, 31))
        self.label_text.setStyleSheet("\n"
"font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_text.setObjectName("label_text")
        self.textNews = QtWidgets.QTextEdit(self.tab_2)
        self.textNews.setGeometry(QtCore.QRect(30, 220, 511, 261))
        self.textNews.setStyleSheet("background-color: rgb(229, 229, 229);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Berlin Sans FB Demi\";")
        self.textNews.setObjectName("textNews")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(30, 500, 511, 31))
        self.pushButton.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton.setObjectName("pushButton")
        self.label_result = QtWidgets.QLabel(self.tab_2)
        self.label_result.setGeometry(QtCore.QRect(250, 540, 71, 21))
        self.label_result.setStyleSheet("font: 75 20pt \"Berlin Sans FB Demi\";")
        self.label_result.setText("")
        self.label_result.setWordWrap(True)
        self.label_result.setObjectName("label_result")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(100, 0, 381, 171))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("resourse/SamaraSAULogo.png"))
        self.label_2.setObjectName("label_2")
        self.pushButton_test = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_test.setGeometry(QtCore.QRect(250, 250, 75, 23))
        self.pushButton_test.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton_test.setObjectName("pushButton_test")
        self.label_accuracy = QtWidgets.QLabel(self.tab_3)
        self.label_accuracy.setGeometry(QtCore.QRect(230, 290, 131, 31))
        self.label_accuracy.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_accuracy.setText("")
        self.label_accuracy.setObjectName("label_accuracy")
        self.label_predskazano_pravilno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_pravilno.setGeometry(QtCore.QRect(200, 320, 181, 21))
        self.label_predskazano_pravilno.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_predskazano_pravilno.setText("")
        self.label_predskazano_pravilno.setObjectName("label_predskazano_pravilno")
        self.label_pravilno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_true_result.setGeometry(QtCore.QRect(210, 350, 161, 21))
        self.label_pravilno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_pravilno_count_true_result.setText("")
        self.label_pravilno_count_true_result.setObjectName("label_pravilno_count_true_result")
        self.label_pravilno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_fake_result.setGeometry(QtCore.QRect(210, 380, 161, 21))
        self.label_pravilno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_pravilno_count_fake_result.setText("")
        self.label_pravilno_count_fake_result.setObjectName("label_pravilno_count_fake_result")
        self.label_neverno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_true_result.setGeometry(QtCore.QRect(220, 440, 161, 21))
        self.label_neverno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_neverno_count_true_result.setText("")
        self.label_neverno_count_true_result.setObjectName("label_neverno_count_true_result")
        self.label_predskazano_neverno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_neverno.setGeometry(QtCore.QRect(210, 410, 181, 21))
        self.label_predskazano_neverno.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_predskazano_neverno.setText("")
        self.label_predskazano_neverno.setObjectName("label_predskazano_neverno")
        self.label_neverno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_fake_result.setGeometry(QtCore.QRect(220, 470, 161, 21))
        self.label_neverno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_neverno_count_fake_result.setText("")
        self.label_neverno_count_fake_result.setObjectName("label_neverno_count_fake_result")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(80, 170, 411, 21))
        self.label_9.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setGeometry(QtCore.QRect(110, 190, 381, 16))
        self.label_10.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab_3)
        self.label_11.setGeometry(QtCore.QRect(170, 210, 221, 16))
        self.label_11.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_3)
        self.label_12.setGeometry(QtCore.QRect(170, 560, 301, 16))
        self.label_12.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_12.setObjectName("label_12")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        machineLearning = MachineLearning(self.label_accuracy, self.label_predskazano_pravilno, self.label_pravilno_count_true_result,
                                          self.label_pravilno_count_fake_result,
                                          self.label_predskazano_neverno, self.label_neverno_count_true_result,
                                          self.label_neverno_count_fake_result, self.label_result)

        self.add_functions(machineLearning)
        self.add_functions_test(machineLearning)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fake news detector"))
        self.label.setText(_translate("MainWindow", "РАСПОЗНАВАТЕЛЬ ФЕЙКОВЫХ НОВОСТЕЙ"))
        self.label_title.setText(_translate("MainWindow", "ЗАГОЛОВОК"))
        self.label_text.setText(_translate("MainWindow", "ТЕКСТ"))
        self.pushButton.setText(_translate("MainWindow", "ПРЕДСКАЗАТЬ"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Распознаватель"))
        self.pushButton_test.setText(_translate("MainWindow", "ТЕСТ"))
        self.label_9.setText(_translate("MainWindow", "Данное программное средство \"Распознаватель фейковых новостей\""))
        self.label_10.setText(_translate("MainWindow", "предназначено для определения правдивости введенного"))
        self.label_11.setText(_translate("MainWindow", " пользователем новостного события."))
        self.label_12.setText(_translate("MainWindow", "© Самарский университет, Калинин А.А. 2021"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "О программе"))




    def add_functions(self, machineLearning):
        self.machineLearning = machineLearning
        self.pushButton.clicked.connect(
            lambda: self.machineLearning.result(self.textTitleNews.toPlainText(), self.textNews.toPlainText()))

    def add_functions_test(self, machineLearning):

        self.machineLearning = machineLearning
        self.pushButton_test.clicked.connect(
            lambda: self.machineLearning.result_test())

    def test(self):
            print(self.textTitleNews.toPlainText())
            print(self.textNews.toPlainText())



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
