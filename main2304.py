import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sqlite3
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(759, 723)
        MainWindow.setStyleSheet("background-color: rgb(229, 229, 229);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 761, 731))
        self.tabWidget.setStyleSheet("background-color: rgb(167, 167, 167);\n"
"border-color: rgb(0, 0, 0);")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(270, 20, 231, 31))
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
        self.label_title.setGeometry(QtCore.QRect(350, 90, 81, 16))
        self.label_title.setStyleSheet("\n"
"font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_title.setObjectName("label_title")
        self.textTitleNews = QtWidgets.QTextEdit(self.tab_2)
        self.textTitleNews.setGeometry(QtCore.QRect(30, 110, 701, 61))
        self.textTitleNews.setStyleSheet("background-color: rgb(229, 229, 229);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Berlin Sans FB Demi\";")
        self.textTitleNews.setObjectName("textTitleNews")
        self.label_text = QtWidgets.QLabel(self.tab_2)
        self.label_text.setGeometry(QtCore.QRect(370, 190, 41, 31))
        self.label_text.setStyleSheet("\n"
"font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_text.setObjectName("label_text")
        self.textNews = QtWidgets.QTextEdit(self.tab_2)
        self.textNews.setGeometry(QtCore.QRect(30, 220, 701, 341))
        self.textNews.setStyleSheet("background-color: rgb(229, 229, 229);\n"
"border-color: rgb(255, 255, 255);\n"
"font: 75 8pt \"Berlin Sans FB Demi\";")
        self.textNews.setObjectName("textNews")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(30, 600, 701, 31))
        self.pushButton.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton.setObjectName("pushButton")
        self.label_result = QtWidgets.QLabel(self.tab_2)
        self.label_result.setGeometry(QtCore.QRect(350, 650, 71, 21))
        self.label_result.setStyleSheet("font: 75 20pt \"Berlin Sans FB Demi\";")
        self.label_result.setWordWrap(True)
        self.label_result.setObjectName("label_result")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tableWidget = QtWidgets.QTableWidget(self.tab)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 761, 681))
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
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(210, 0, 381, 171))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("resourse/SamaraSAULogo.png"))
        self.label_2.setObjectName("label_2")
        self.pushButton_test = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_test.setGeometry(QtCore.QRect(20, 290, 75, 23))
        self.pushButton_test.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton_test.setObjectName("pushButton_test")
        self.label_accuracy = QtWidgets.QLabel(self.tab_3)
        self.label_accuracy.setGeometry(QtCore.QRect(20, 330, 131, 31))
        self.label_accuracy.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_accuracy.setObjectName("label_accuracy")
        self.label_predskazano_pravilno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_pravilno.setGeometry(QtCore.QRect(20, 360, 181, 21))
        self.label_predskazano_pravilno.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_predskazano_pravilno.setObjectName("label_predskazano_pravilno")
        self.label_pravilno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_true_result.setGeometry(QtCore.QRect(20, 380, 161, 21))
        self.label_pravilno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_pravilno_count_true_result.setObjectName("label_pravilno_count_true_result")
        self.label_pravilno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_fake_result.setGeometry(QtCore.QRect(20, 400, 161, 21))
        self.label_pravilno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_pravilno_count_fake_result.setObjectName("label_pravilno_count_fake_result")
        self.label_neverno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_true_result.setGeometry(QtCore.QRect(20, 460, 161, 21))
        self.label_neverno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_neverno_count_true_result.setObjectName("label_neverno_count_true_result")
        self.label_predskazano_neverno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_neverno.setGeometry(QtCore.QRect(20, 440, 181, 21))
        self.label_predskazano_neverno.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_predskazano_neverno.setObjectName("label_predskazano_neverno")
        self.label_neverno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_fake_result.setGeometry(QtCore.QRect(20, 480, 161, 21))
        self.label_neverno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_neverno_count_fake_result.setObjectName("label_neverno_count_fake_result")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(180, 160, 411, 21))
        self.label_9.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setGeometry(QtCore.QRect(210, 180, 381, 16))
        self.label_10.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab_3)
        self.label_11.setGeometry(QtCore.QRect(280, 200, 221, 16))
        self.label_11.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_3)
        self.label_12.setGeometry(QtCore.QRect(270, 660, 301, 16))
        self.label_12.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_12.setObjectName("label_12")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(220, 260, 521, 371))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("graphic.png"))
        self.label_3.setObjectName("label_3")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.add_functions()
        self.add_functions_test()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Fake news detector"))
        self.label.setText(_translate("MainWindow", "FAKE NEWS DETECTOR"))
        self.label_title.setText(_translate("MainWindow", "ЗАГОЛОВОК"))
        self.label_text.setText(_translate("MainWindow", "ТЕКСТ"))
        self.pushButton.setText(_translate("MainWindow", "ПРЕДСКАЗАТЬ"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Распознаватель"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Текст"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Результат"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Дата"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "История"))
        self.pushButton_test.setText(_translate("MainWindow", "ТЕСТ"))
        self.label_9.setText(_translate("MainWindow", "Данное программное средство \"Распознаватель фейковых новостей\""))
        self.label_10.setText(_translate("MainWindow", "предназначено для определения правдивости введенного"))
        self.label_11.setText(_translate("MainWindow", " пользователем новостного события."))
        self.label_12.setText(_translate("MainWindow", "© Самарский университет, Калинин А.А. 2021"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "О программе"))

        self.tableWidget.setColumnWidth(0, 100)
        self.tableWidget.setColumnWidth(1, 240)
        self.tableWidget.setColumnWidth(2, 75)
        self.tableWidget.setColumnWidth(3, 100)

        self.loaddata()

    def loaddata(self):
        connection = sqlite3.connect('resourse/history.db')
        cursor = connection.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS news ("title" TEXT,"text" TEXT,"result" TEXT, "data" TEXT)""")
        sqlShowData = "SELECT * FROM news LIMIT 50"

        self.tableWidget.setRowCount(50)
        tableRow = 0
        for row in cursor.execute(sqlShowData):
            self.tableWidget.setItem(tableRow, 0, QtWidgets.QTableWidgetItem(row[0]))
            self.tableWidget.setItem(tableRow, 1, QtWidgets.QTableWidgetItem(row[1]))
            self.tableWidget.setItem(tableRow, 2, QtWidgets.QTableWidgetItem(row[2]))
            self.tableWidget.setItem(tableRow, 3, QtWidgets.QTableWidgetItem(row[3]))
            tableRow+=1


    def add_functions(self):
        self.pushButton.clicked.connect(
            lambda: self.result(self.textTitleNews.toPlainText(), self.textNews.toPlainText()))

    def add_functions_test(self):
        self.pushButton_test.clicked.connect(
            lambda: self.result_test())

    def test(self):
            print(self.textTitleNews.toPlainText())
            print(self.textNews.toPlainText())

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

        #self.drawGraphic(temp111,temp122,temp112,temp121)
        self.drawGraphicTest()



    def drawGraphic(self, predictedTrueTrue, predictedTrueFake, predictedFalseTrue, predictedFalseFake):
        # !/usr/bin/env python3
        # vim: set ai et ts=4 sw=4:

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import datetime as dt
        import csv

        data_names = ['true', 'true1', 'false', 'false1']
        #data_values = [predictedTrueTrue, predictedFalseTrue, predictedTrueFake, predictedFalseFake]
        data_values = [predictedTrueTrue, predictedTrueFake, predictedFalseTrue, predictedFalseFake]

        dpi = 80
        fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
        mpl.rcParams.update({'font.size': 10})

        plt.title('Диаграмма полученных данных')

        ax = plt.axes()
        ax.yaxis.grid(True, zorder=1)

        xs = range(len(data_names))

        plt.bar([x + 0.05 for x in xs], [d * 0.9 for d in data_values],
                width=0.2, color='blue', alpha=0.7, label='верно',
                zorder=2)
        plt.bar([x + 0.3 for x in xs], data_values,
                width=0.2, color='red', alpha=0.7, label='неверно',
                zorder=2)
        plt.xticks(xs, data_names)

        fig.autofmt_xdate(rotation=25)

        plt.legend(loc='upper right')
        fig.savefig('graphic.png')
        self.label_3.setPixmap(QtGui.QPixmap("graphic.png"))

    def drawGraphicTest(self):
        cat_par = [f"P{i}" for i in range(5)]
        g1 = [10, 21, 34, 12, 27]
        g2 = [17, 15, 25, 21, 26]
        width = 0.3
        x = np.arange(len(cat_par))
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, g1, width, label='g1')
        rects2 = ax.bar(x + width / 2, g2, width, label='g2')
        ax.set_title('Пример групповой диаграммы')
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
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)  # Матрица фукнций (стоп-слово, 0.7 - автоматическое обнаружение стоп-слова)
        # DataFlair - Fit and transform train set, transform test set
        tfidf_train = tfidf_vectorizer.fit_transform(x_train)  # центрирование обучающего набора
        tfidf_test = tfidf_vectorizer.transform(x_test_in)  # перемешивание данных

        # DataFlair - Initialize a PassiveAggressiveClassifier
        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train, y_train)
        # DataFlair - Predict on the test set and calculate accuracy
        y_pred = pac.predict(tfidf_test)

        print("полученные лейблы:")
        print(y_pred)
        result = list(y_pred)
        for i in range(len(result)):
            self.label_result.setText(result[i])

        connection = sqlite3.connect('resourse/history.db')
        cursor = connection.cursor()
        now = datetime.datetime.now()
        currentDate = now.strftime("%d-%m-%Y %H:%M")
        print(currentDate)
        cursor.execute("""CREATE TABLE IF NOT EXISTS news ("title" TEXT,"text" TEXT, "result" TEXT, "data" TEXT)""")
        sqlAddData = f"INSERT INTO news VALUES ('{title}', '{text}','{result[i]}','{currentDate}')"
        cursor.execute(sqlAddData)
        connection.commit()
        self.loaddata()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
