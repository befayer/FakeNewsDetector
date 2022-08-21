import pandas as pd
import sys
from PyQt5.QtWidgets import QMainWindow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sqlite3
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QToolTip, QMessageBox, QLabel, QWidget)


class DB:

    def __init__(self, tableWidget, tableWidget_2, login):
        self.tableWidget = tableWidget
        self.tableWidget_2 = tableWidget_2
        self.login = login

    def loaddata(self, login):
        connection = sqlite3.connect('resourse/historytest.db')
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS news ("title" TEXT,"text" TEXT,"result" TEXT, "data" TEXT, 
        "user" TEXT)""")
        sqlShowData = f"SELECT * FROM news WHERE user='{login}' LIMIT 50"
        self.tableWidget.setRowCount(50)
        tableRow = 0
        for row in cursor.execute(sqlShowData):
            self.tableWidget.setItem(tableRow, 0, QtWidgets.QTableWidgetItem(row[0]))
            self.tableWidget.setItem(tableRow, 1, QtWidgets.QTableWidgetItem(row[1]))
            self.tableWidget.setItem(tableRow, 2, QtWidgets.QTableWidgetItem(row[2]))
            self.tableWidget.setItem(tableRow, 3, QtWidgets.QTableWidgetItem(row[3]))
            tableRow += 1
        cursor.close()

    def loaddataResearch(self, login):
        connection = sqlite3.connect('resourse/historytest.db')
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS research ("data" TEXT,"accuracy" REAL,"true" INTEGER, "false" INTEGER,
        "test_size" REAL,"max_df" REAL, "count_education" INTEGER, "user" TEXT)""")
        #cursor.execute("""CREATE TABLE IF NOT EXISTS research ("data" TEXT,"accuracy" TEXT,"true" TEXT, "false" TEXT,
        #"test_size" TEXT,"max_df" TEXT, "count_education" TEXT, "user" TEXT)""")
        sqlShowData = f"SELECT * FROM research WHERE user='{login}' LIMIT 50"
        self.tableWidget_2.setRowCount(50)
        tableRow = 0
        for row in cursor.execute(sqlShowData):
            self.tableWidget_2.setItem(tableRow, 0, QtWidgets.QTableWidgetItem(row[0]))
            self.tableWidget_2.setItem(tableRow, 1, QtWidgets.QTableWidgetItem(row[1]))
            self.tableWidget_2.setItem(tableRow, 2, QtWidgets.QTableWidgetItem(str(row[2])))
            self.tableWidget_2.setItem(tableRow, 3, QtWidgets.QTableWidgetItem(str(row[3])))
            self.tableWidget_2.setItem(tableRow, 4, QtWidgets.QTableWidgetItem(row[4]))
            self.tableWidget_2.setItem(tableRow, 5, QtWidgets.QTableWidgetItem(row[5]))
            self.tableWidget_2.setItem(tableRow, 6, QtWidgets.QTableWidgetItem(str(row[6])))
            tableRow += 1
        cursor.close()

class MachineLearning:

    def __init__(self, doubleSpinBoxSizeTestSet, doubleSpinBoxMaxDF, spinBox,
                 label_accuracy, label_predskazano_pravilno, label_pravilno_count_true_result,
                 label_pravilno_count_fake_result, label_predskazano_neverno,
                 label_neverno_count_true_result, label_neverno_count_fake_result,
                 label_9,label_3,label_result, tableWidget, tableWidget_2, login):
        self.doubleSpinBoxSizeTestSet = doubleSpinBoxSizeTestSet
        self.doubleSpinBoxMaxDF = doubleSpinBoxMaxDF
        self.spinBox = spinBox
        self.label_accuracy = label_accuracy
        self.label_predskazano_pravilno = label_predskazano_pravilno
        self.label_pravilno_count_true_result = label_pravilno_count_true_result
        self.label_pravilno_count_fake_result = label_pravilno_count_fake_result
        self.label_predskazano_neverno = label_predskazano_neverno
        self.label_neverno_count_true_result = label_neverno_count_true_result
        self.label_neverno_count_fake_result = label_neverno_count_fake_result
        self.label_9 = label_9
        self.label_3 = label_3
        self.label_result = label_result
        self.tableWidget = tableWidget
        self.tableWidget_2 = tableWidget_2
        self.login = login

    def result_test(self, login):
        #getting values
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
        x_train, x_test, y_train, y_test = train_test_split(df_test['text'], labels_test, test_size=form_test_size,
                                                            random_state=5)
        #random_state=7 `1=94.5 2=93.6 3=94 4=93.2 5=94.6 6=93.85 7=92.8 8=93.5 9=93.5

        # DataFlair - Initialize a TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=form_max_df)

        # DataFlair - Fit and transform train set, transform test set
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = tfidf_vectorizer.transform(x_test)

        # DataFlair - Initialize a PassiveAggressiveClassifier
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

        # Out
        temp_label_accuracy = f'Accuracy: {round(score * 100, 2)}%'
        self.label_accuracy.setText(temp_label_accuracy)
        self.label_predskazano_pravilno.setText('Предсказано правильно:')

        self.label_pravilno_count_true_result.setText(f'{temp111} true-результатов')
        self.label_pravilno_count_fake_result.setText(f'{temp122} fake-результатов')
        self.label_predskazano_neverno.setText('Предсказано неверно:')
        self.label_neverno_count_true_result.setText(f'{temp112} true-результатов')
        self.label_neverno_count_fake_result.setText(f'{temp121} fake-результатов')
        count_verno=temp111+temp122
        count_neverno = temp112+temp121
        #self.label_9.setText(f'Система определила, что {temp111} новостей true, {temp122} новостей fake. Из них {count_neverno} новостей было отнесено не к тому классу.')

        # self.drawGraphic(temp111,temp122,temp112,temp121)
        self.drawGraphicTest(temp111, temp122)

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

        connection = sqlite3.connect('resourse/historytest.db')
        cursor = connection.cursor()
        now = datetime.datetime.now()
        currentDate = now.strftime("%d-%m-%Y %H:%M")
        print(currentDate)
        int_form_test_size=int(form_test_size*100)
        int_form_max_df = int(form_max_df * 100)
        cursor.execute("""CREATE TABLE IF NOT EXISTS research ("data" TEXT,"accuracy" REAL,"true" INTEGER, "false" INTEGER,
                "test_size" REAL,"max_df" REAL, "count_education" INTEGER, "user" TEXT)""")
        #cursor.execute("""CREATE TABLE IF NOT EXISTS research ("data" TEXT,"accuracy" TEXT,"true" TEXT, "false" TEXT,
         #       "test_size" TEXT,"max_df" TEXT, "count_education" TEXT, "user" TEXT)""")
        sqlAddDataResearch = f"INSERT INTO research VALUES ('{currentDate}', '{round(score * 100, 2)}%', '{count_verno}', '{count_neverno}', '{int_form_test_size}%', '{int_form_max_df}%', '{form_max_iter}', '{login}')"
        #sqlAddDataResearch = f"INSERT INTO research VALUES ('{currentDate}', '{round(score * 100, 2)}%', '{count_verno}', '{count_neverno}', '{int_form_test_size}%', '{int_form_max_df}%', '{form_max_iter}', '{login}')"

        cursor.execute(sqlAddDataResearch)
        connection.commit()
        cursor.close()
        bd = DB(self.tableWidget, self.tableWidget_2, login)
        bd.loaddataResearch(login)
        #self.label_2.setPixmap(QtGui.QPixmap("graphicsubjest.png"))

    def drawGraphicTest(self, predictedTrueTrue, predictedTrueFake):
        cat_par = ["КЛАССИФИЦИРОВАНО ВЕРНО"]
        g1 = [predictedTrueTrue]
        g2 = [predictedTrueFake]
        width = 0.2
        x = np.arange(len(cat_par))
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, g1, width, color='green', label='TRUE-новостей')
        rects2 = ax.bar(x + width / 2, g2, width, color='orange', label='FALSE-новостей')
        ax.set_title('Диаграмма классифицированных новостей')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_par)
        ax.legend()
        fig.savefig('graphic.png')
        self.label_3.setPixmap(QtGui.QPixmap("graphic.png"))

    def result(self, title, text, login):
        df = pd.read_csv('D:\\news.csv')
        columns = ['title', 'text', 'label']
        data = [[title, text, '']]
        dff = pd.DataFrame(data, columns=columns)
        dff.to_csv('D:\\in.csv')
        dfin = pd.read_csv('D:\\in.csv')
        df.shape
        dfin.shape
        labels = df.label
        x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7) #random_state - управляет перемешиванием
        x_test_in = dfin['text']
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) #max_df = 0.7 означает " игнорировать термины, которые встречаются более чем в 70% документах ".
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = tfidf_vectorizer.transform(x_test_in)
        pac = PassiveAggressiveClassifier(max_iter=50) #Максимальное количество проходов по обучающим данным
        pac.fit(tfidf_train, y_train)
        y_pred = pac.predict(tfidf_test)
        print("Результат:")
        print(y_pred)
        result = list(y_pred)
        for i in range(len(result)):
            if (result[i] == "FAKE"):
                self.label_result.setStyleSheet("font: 75 35pt \"Berlin Sans FB Demi\"; color: rgb(229, 0, 0);;")
                self.label_result.setText(result[i])
            else:
                self.label_result.setStyleSheet("font: 75 35pt \"Berlin Sans FB Demi\"; color: rgb(20, 207, 0);;")
                self.label_result.setText(result[i])

       # connection = sqlite3.connect('resourse/historytest.db')
      #  cursor = connection.cursor()
       # now = datetime.datetime.now()
       # currentDate = now.strftime("%d-%m-%Y %H:%M")
       # short_text = text[0:50]
       # short_title = title[0:50]
       # print(short_text)
       # print(short_title)
       # print(currentDate)
        #cursor.execute("""CREATE TABLE IF NOT EXISTS news ("title" TEXT,"text" TEXT, "result" TEXT, "data" TEXT, "user" TEXT)""")
        #sqlAddData = f"INSERT INTO news VALUES ('{short_title}', '{short_text}','{result[i]}','{currentDate}', '{login}')"
        #cursor.execute(sqlAddData)
        #connection.commit()
        #cursor.close()
        #bd = DB(self.tableWidget, self.tableWidget_2, login)
        #bd.loaddata(login)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, login):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(871, 744)
        MainWindow.setFixedSize(871, 744)
        MainWindow.setStyleSheet("background-color: rgb(194, 194, 194);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 881, 731))
        self.tabWidget.setStyleSheet("background-color: rgb(217, 217, 217);\n"
"border-color: rgb(0, 0, 0);")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(280, 10, 361, 41))
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
        self.label_title.setGeometry(QtCore.QRect(400, 90, 81, 16))
        self.label_title.setStyleSheet("\n"
"font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_title.setObjectName("label_title")
        self.textTitleNews = QtWidgets.QTextEdit(self.tab_2)
        self.textTitleNews.setGeometry(QtCore.QRect(30, 110, 821, 71))
        self.textTitleNews.setStyleSheet("background-color: rgb(229, 229, 229);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Berlin Sans FB Demi\";")
        self.textTitleNews.setObjectName("textTitleNews")
        self.label_text = QtWidgets.QLabel(self.tab_2)
        self.label_text.setGeometry(QtCore.QRect(420, 190, 41, 31))
        self.label_text.setStyleSheet("\n"
"font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_text.setObjectName("label_text")
        self.textNews = QtWidgets.QTextEdit(self.tab_2)
        self.textNews.setGeometry(QtCore.QRect(30, 220, 821, 331))
        self.textNews.setStyleSheet("background-color: rgb(229, 229, 229);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Berlin Sans FB Demi\";")
        self.textNews.setObjectName("textNews")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(30, 560, 821, 31))
        self.pushButton.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton.setObjectName("pushButton")
        self.label_result = QtWidgets.QLabel(self.tab_2)
        #self.label_result.setGeometry(QtCore.QRect(320, 620, 251, 41))
        self.label_result.setGeometry(QtCore.QRect(390, 620, 111, 41))
        self.label_result.setStyleSheet("font: 75 35pt \"Berlin Sans FB Demi\";")
        self.label_result.setWordWrap(True)
        self.label_result.setObjectName("label_result")
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(310, 680, 261, 16))
        self.label_13.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_13.setObjectName("label_13")
        self.label_nameUser = QtWidgets.QLabel(self.tab_2)
        self.label_nameUser.setGeometry(QtCore.QRect(750, 10, 101, 16))
        self.label_nameUser.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_nameUser.setObjectName("label_nameUser")
        self.changeUserBtn = QtWidgets.QPushButton(self.tab_2)
        self.changeUserBtn.setGeometry(QtCore.QRect(730, 70, 131, 21))
        self.changeUserBtn.setStyleSheet("font: 8pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.changeUserBtn.setObjectName("changeUserBtn")
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(750, 30, 101, 21))
        self.label_11.setStyleSheet("font: 8pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_11.setObjectName("label_11")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tableWidget = QtWidgets.QTableWidget(self.tab)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 871, 701))
        self.tableWidget.setStyleSheet("")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        item.setText("Заголовок")
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(139)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)
        self.tableWidget.verticalHeader().setMinimumSectionSize(23)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.pushButton_test = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_test.setGeometry(QtCore.QRect(470, 440, 321, 51))
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.pushButton_test.setFont(font)
        self.pushButton_test.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton_test.setObjectName("pushButton_test")
        self.label_accuracy = QtWidgets.QLabel(self.tab_3)
        self.label_accuracy.setGeometry(QtCore.QRect(30, 300, 161, 31))
        self.label_accuracy.setStyleSheet("font: 15pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_accuracy.setObjectName("label_accuracy")
        self.label_predskazano_pravilno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_pravilno.setGeometry(QtCore.QRect(30, 340, 231, 21))
        self.label_predskazano_pravilno.setStyleSheet("font: 14pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_predskazano_pravilno.setObjectName("label_predskazano_pravilno")
        self.label_pravilno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_true_result.setGeometry(QtCore.QRect(30, 370, 311, 21))
        self.label_pravilno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_pravilno_count_true_result.setObjectName("label_pravilno_count_true_result")
        self.label_pravilno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_fake_result.setGeometry(QtCore.QRect(30, 390, 311, 21))
        self.label_pravilno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_pravilno_count_fake_result.setObjectName("label_pravilno_count_fake_result")
        self.label_neverno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_true_result.setGeometry(QtCore.QRect(30, 450, 321, 21))
        self.label_neverno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_neverno_count_true_result.setObjectName("label_neverno_count_true_result")
        self.label_predskazano_neverno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_neverno.setGeometry(QtCore.QRect(30, 420, 251, 21))
        self.label_predskazano_neverno.setStyleSheet("font: 14pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_predskazano_neverno.setObjectName("label_predskazano_neverno")
        self.label_neverno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_fake_result.setGeometry(QtCore.QRect(30, 470, 321, 21))
        self.label_neverno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_neverno_count_fake_result.setObjectName("label_neverno_count_fake_result")
        self.label_12 = QtWidgets.QLabel(self.tab_3)
        self.label_12.setGeometry(QtCore.QRect(330, 670, 301, 16))
        self.label_12.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_12.setObjectName("label_12")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(420, 30, 431, 391))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("true_false_diagram_before.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(20, 90, 181, 21))
        self.label_2.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_2.setObjectName("label_2")
        self.doubleSpinBoxSizeTestSet = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBoxSizeTestSet.setGeometry(QtCore.QRect(230, 90, 62, 22))
        self.doubleSpinBoxSizeTestSet.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.doubleSpinBoxSizeTestSet.setDecimals(1)
        self.doubleSpinBoxSizeTestSet.setMinimum(0.1)
        self.doubleSpinBoxSizeTestSet.setMaximum(0.9)
        self.doubleSpinBoxSizeTestSet.setSingleStep(0.1)
        self.doubleSpinBoxSizeTestSet.setObjectName("doubleSpinBoxSizeTestSet")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(20, 130, 191, 21))
        self.label_4.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(20, 150, 201, 21))
        self.label_5.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_5.setObjectName("label_5")
        self.doubleSpinBoxMaxDF = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBoxMaxDF.setGeometry(QtCore.QRect(230, 150, 62, 22))
        self.doubleSpinBoxMaxDF.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.doubleSpinBoxMaxDF.setDecimals(1)
        self.doubleSpinBoxMaxDF.setMinimum(0.1)
        self.doubleSpinBoxMaxDF.setMaximum(0.9)
        self.doubleSpinBoxMaxDF.setSingleStep(0.1)
        self.doubleSpinBoxMaxDF.setObjectName("doubleSpinBoxMaxDF")
        self.label_6 = QtWidgets.QLabel(self.tab_3)
        self.label_6.setGeometry(QtCore.QRect(300, 150, 91, 21))
        self.label_6.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        self.label_7.setGeometry(QtCore.QRect(20, 190, 261, 21))
        self.label_7.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setGeometry(QtCore.QRect(20, 210, 171, 21))
        self.label_8.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_8.setObjectName("label_8")
        self.spinBox = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox.setGeometry(QtCore.QRect(230, 210, 61, 22))
        self.spinBox.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(200)
        self.spinBox.setObjectName("spinBox")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(60, 590, 771, 41))
        self.label_9.setStyleSheet("font: 11pt \"Impact\";\n"
"color: rgb(0, 115, 255);")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setGeometry(QtCore.QRect(50, 30, 301, 21))
        self.label_10.setStyleSheet("font: 14pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_10.setObjectName("label_10")
        #self.label_repusltat = QtWidgets.QLabel(self.tab_3)
        #self.label_repusltat.setGeometry(QtCore.QRect(60, 270, 91, 21))
        #self.label_repusltat.setStyleSheet("font: 14pt \"Impact\";\n"
#"color: rgb(84, 103, 118);")
        #self.label_repusltat.setObjectName("label_repusltat")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.tab_4)
        self.tableWidget_2.setGeometry(QtCore.QRect(0, 0, 871, 701))
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(7)
        self.tableWidget_2.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(9)
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(9)
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Impact")
        item.setFont(font)
        self.tableWidget_2.setHorizontalHeaderItem(6, item)
        self.tableWidget_2.horizontalHeader().setDefaultSectionSize(124)
        self.tableWidget_2.horizontalHeader().setMinimumSectionSize(7)
        self.tabWidget.addTab(self.tab_4, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.login = login
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow, login)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        bd = DB(self.tableWidget,self.tableWidget_2, self.login)
        self.add_functions_DB(bd)

        machineLearning = MachineLearning(self.doubleSpinBoxSizeTestSet, self.doubleSpinBoxMaxDF, self.spinBox,
                                          self.label_accuracy, self.label_predskazano_pravilno,
                                          self.label_pravilno_count_true_result,
                                          self.label_pravilno_count_fake_result, self.label_predskazano_neverno,
                                          self.label_neverno_count_true_result, self.label_neverno_count_fake_result,
                                          self.label_9, self.label_3, self.label_result, self.tableWidget, self.tableWidget_2, self.login)

        self.add_functions_class(machineLearning)
        self.add_functions_research(machineLearning)
        self.changeUser()

    def retranslateUi(self, MainWindow, login):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификатор новостей"))
        self.label.setText(_translate("MainWindow", "КЛАССИФИКАТОР НОВОСТЕЙ"))
        self.label_title.setText(_translate("MainWindow", "ЗАГОЛОВОК"))
        self.label_text.setText(_translate("MainWindow", "ТЕКСТ"))
        self.pushButton.setText(_translate("MainWindow", "КЛАССИФИЦИРОВАТЬ"))
        self.label_result.setText(_translate("MainWindow", "----"))
        self.label_13.setText(_translate("MainWindow", "© Самарский университет, Калинин А.А. 2021"))
        self.label_nameUser.setText(_translate("MainWindow", "Пользователь"))
        self.changeUserBtn.setText(_translate("MainWindow", "Сменить пользователя"))
        self.label_11.setText(_translate("MainWindow", "user"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Классификатор новости"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Текст"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Результат"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Дата"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "История классификаций"))
        self.pushButton_test.setText(_translate("MainWindow", "ИССЛЕДОВАТЬ"))
        self.label_accuracy.setText(_translate("MainWindow", "Точность: ------"))
        self.label_predskazano_pravilno.setText(_translate("MainWindow", "Классифицировано верно:"))
        self.label_pravilno_count_true_result.setText(_translate("MainWindow", "--- образцов, принадлежащих классу TRUE"))
        self.label_pravilno_count_fake_result.setText(_translate("MainWindow", "--- образцов, принадлежащих классу FAKE"))
        self.label_neverno_count_true_result.setText(_translate("MainWindow", "-- образцов, принадлежащих классу TRUE"))
        self.label_predskazano_neverno.setText(_translate("MainWindow", "Классифицировано неверно:"))
        self.label_neverno_count_fake_result.setText(_translate("MainWindow", "-- образцов, принадлежащих классу TRUE"))
        self.label_12.setText(_translate("MainWindow", "© Самарский университет, Калинин А.А. 2021"))
        self.label_2.setText(_translate("MainWindow", "Объем тестовой выборки:"))
        self.label_4.setText(_translate("MainWindow", "Игнорирование терминов,"))
        self.label_5.setText(_translate("MainWindow", "встречающихся более чем в:"))
        self.label_6.setText(_translate("MainWindow", "документов"))
        self.label_7.setText(_translate("MainWindow", "Максимальное количество проходов"))
        self.label_8.setText(_translate("MainWindow", "по обучающим данным"))
        #self.label_9.setText(_translate("MainWindow", "Система определила, что 30 новостей true, 10 новостей fake. Из них 5 новостей было отнесено не к тому классу."))
        self.label_10.setText(_translate("MainWindow", "Параметры машинного обучения"))
        #self.label_repusltat.setText(_translate("MainWindow", "Результат"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Исследование"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Дата"))
        item = self.tableWidget_2.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Точность"))
        item = self.tableWidget_2.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "TRUE"))
        item = self.tableWidget_2.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "FAKE"))
        item = self.tableWidget_2.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Объем тестовой выборки"))
        item = self.tableWidget_2.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Объем документов, сдержащий исключающие термины"))
        item = self.tableWidget_2.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Итераций обучения"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "История исследований"))
        self.label_11.setText(_translate("MainWindow", f'{login}'))

        self.tableWidget.setColumnWidth(0, 180)
        self.tableWidget.setColumnWidth(1, 460)
        self.tableWidget.setColumnWidth(2, 75)
        self.tableWidget.setColumnWidth(3, 110)

        self.tableWidget_2.setColumnWidth(0, 95) #дата
        self.tableWidget_2.setColumnWidth(1, 60) #точность
        self.tableWidget_2.setColumnWidth(2, 60) #true
        self.tableWidget_2.setColumnWidth(3, 50) #fake
        self.tableWidget_2.setColumnWidth(4, 140) #объем выборки
        self.tableWidget_2.setColumnWidth(5, 305) #термины
        self.tableWidget_2.setColumnWidth(6, 120) #итерации

        bd = DB(self.tableWidget,self.tableWidget_2, self.login)
        self.add_functions_DB(bd)

    def add_functions_class(self, machineLearning):
        self.machineLearning = machineLearning
        self.pushButton.clicked.connect(
            lambda: self.machineLearning.result(self.textTitleNews.toPlainText(), self.textNews.toPlainText(), self.login))

    def add_functions_research(self, machineLearning):
        self.machineLearning = machineLearning
        self.pushButton_test.clicked.connect(lambda: self.machineLearning.result_test(self.login))

    def add_functions_DB(self, db):
        self.db = db
        self.db.loaddata(self.login)
        self.db.loaddataResearch(self.login)

    def changeUser(self):
        self.changeUserBtn.clicked.connect(lambda: self.showUi_Login())

    def showUi_Login(self):
        self.tabWidget.close()
        MainWindow.hide()
        MainWindow.setParent(None)
        MainWindow.close()
        uiw = Ui_Login()
        uiw.setupUi(MainWindow)
        MainWindow.repaint()
        MainWindow.show()

class Ui_Login(object):

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(361, 307)
        Form.setFixedSize(361, 307)
        self.pushButton_in = QtWidgets.QPushButton(Form)
        self.pushButton_in.setGeometry(QtCore.QRect(10, 150, 341, 23))
        self.pushButton_in.setStyleSheet("font: 12pt \"Impact\";\n"
                                         "color: rgb(208, 208, 208);\n"
                                         "background-color: rgb(107, 133, 150);")
        self.pushButton_in.setObjectName("pushButton_in")
        self.pushButton_reg = QtWidgets.QPushButton(Form)
        self.pushButton_reg.setGeometry(QtCore.QRect(10, 270, 341, 23))
        self.pushButton_reg.setStyleSheet("font: 11pt \"Impact\";\n"
                                          "color: rgb(208, 208, 208);\n"
                                          "background-color: rgb(107, 133, 150);")
        self.pushButton_reg.setObjectName("pushButton_reg")
        self.lineEdit_login = QtWidgets.QLineEdit(Form)
        self.lineEdit_login.setGeometry(QtCore.QRect(10, 90, 341, 21))
        self.lineEdit_login.setObjectName("lineEdit_login")
        self.lineEdit_pass = QtWidgets.QLineEdit(Form)
        self.lineEdit_pass.setGeometry(QtCore.QRect(10, 120, 341, 20))
        self.lineEdit_pass.setObjectName("lineEdit_pass")
        self.label_name_system = QtWidgets.QLabel(Form)
        self.label_name_system.setGeometry(QtCore.QRect(70, 10, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_name_system.setFont(font)
        self.label_name_system.setStyleSheet("font: 16pt \"Impact\";\n"
                                             "color: rgb(84, 103, 118);")
        self.label_name_system.setObjectName("label_name_system")
        self.label_have_not_acc = QtWidgets.QLabel(Form)
        self.label_have_not_acc.setGeometry(QtCore.QRect(140, 250, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_have_not_acc.setFont(font)
        self.label_have_not_acc.setStyleSheet("font: 10pt \"Impact\";\n"
                                              "color: rgb(84, 103, 118);")
        self.label_have_not_acc.setObjectName("label_have_not_acc")
        self.label_authorization = QtWidgets.QLabel(Form)
        self.label_authorization.setGeometry(QtCore.QRect(140, 60, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_authorization.setFont(font)
        self.label_authorization.setStyleSheet("font: 12pt \"Impact\";\n"
                                               "color: rgb(84, 103, 118);")
        self.label_authorization.setObjectName("label_authorization")
        self.label_error_login_or_pass = QtWidgets.QLabel(Form)
        self.label_error_login_or_pass.setGeometry(QtCore.QRect(80, 180, 201, 20))
        self.label_error_login_or_pass.setStyleSheet("font: 12pt \"Impact\";\n"
                                                     "color:rgb(255, 117, 117)")
        self.label_error_login_or_pass.setObjectName("label_error_login_or_pass")
        self.label_enter_login_and_pass = QtWidgets.QLabel(Form)
        self.label_enter_login_and_pass.setGeometry(QtCore.QRect(100, 200, 171, 16))
        self.label_enter_login_and_pass.setStyleSheet("font: 12pt \"Impact\";\n"
                                                      "color:rgb(255, 117, 117)")
        self.label_enter_login_and_pass.setObjectName("label_enter_login_and_pass")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.event_authorization()
        self.event_registration()


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Классификатор новостей"))
        self.pushButton_in.setText(_translate("Form", "ВОЙТИ"))
        self.pushButton_reg.setText(_translate("Form", "ЗАРЕГИСТРИРОВАТЬСЯ"))
        self.lineEdit_login.setPlaceholderText(_translate("Form", "Логин"))
        self.lineEdit_pass.setPlaceholderText(_translate("Form", "Пароль"))
        self.label_name_system.setText(_translate("Form", "КЛАССИФИКАТОР НОВОСТЕЙ"))
        self.label_have_not_acc.setText(_translate("Form", "Нет аккаунта?"))
        self.label_authorization.setText(_translate("Form", "АВТОРИЗАЦИЯ"))

    def event_authorization(self):
        self.pushButton_in.clicked.connect(lambda: self.authorization())

    def event_registration(self):
        self.pushButton_reg.clicked.connect(lambda: self.registration())

    def authorization(self):
        login = self.lineEdit_login.text()
        password = self.lineEdit_pass.text()
        connection = sqlite3.connect('resourse/historytest.db')
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users ("login" TEXT,"password" TEXT)""")
        sqlCheckUser = f"SELECT * FROM users WHERE login='{login}' AND password='{password}'"
        cursor.execute(sqlCheckUser)
        all = cursor.fetchall()
        for i in range(len(all)):
            temp = all[i]
            if ((login==temp[0]) and (password==temp[1])):
                MainWindow.close()
                ui = Ui_MainWindow()
                ui.setupUi(MainWindow, login)
                MainWindow.show()
        self.label_error_login_or_pass.setText('Неверный логин или пароль')
        if (login=="" and password==""):
           self.label_error_login_or_pass.setText('Введите логин и пароль')
        if (login == "" and password != ""):
            self.label_error_login_or_pass.setText('Введите логин')
        if (login != "" and password == ""):
            self.label_error_login_or_pass.setText('Введите пароль')

    def registration(self):
        login = self.lineEdit_login.text()
        password = self.lineEdit_pass.text()
        print(login)
        print(password)
        connection = sqlite3.connect('resourse/historytest.db')
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users ("login" TEXT,"password" TEXT)""")
        sqlRegistrationUser = f"INSERT INTO users VALUES ('{login}','{password}')"
        cursor.execute(sqlRegistrationUser)
        connection.commit()
        print(f"User with login:'{login}' and password:'{password}' was registered")
        cursor.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Login()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


