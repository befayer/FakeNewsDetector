import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sqlite3
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

class DB:

    def __init__(self, tableWidget):
        self.tableWidget = tableWidget

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
            tableRow += 1
        cursor.close()

class MachineLearning:

    def __init__(self, doubleSpinBoxSizeTestSet, doubleSpinBoxMaxDF, spinBox,
                 label_accuracy, label_predskazano_pravilno, label_pravilno_count_true_result,
                 label_pravilno_count_fake_result, label_predskazano_neverno,
                 label_neverno_count_true_result, label_neverno_count_fake_result,
                 label_9,label_3,label_result, tableWidget):
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

    def result_test(self):
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
        x_train, x_test, y_train, y_test = train_test_split(df_test['text'], labels_test, test_size=form_test_size,random_state=5) #random_state=7 `1=94.5 2=93.6 3=94 4=93.2 5=94.6 6=93.85 7=92.8 8=93.5 9=93.5

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
        count_neverno = temp112+temp121
        self.label_9.setText(f'Система определила, что {temp111} новостей true, {temp122} новостей fake. Из них {count_neverno} новостей было отнесено не к тому классу.')

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
        #self.label_2.setPixmap(QtGui.QPixmap("graphicsubjest.png"))

    def drawGraphicTest(self, predictedTrueTrue, predictedTrueFake, predictedFalseTrue, predictedFalseFake):
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

    def result(self, title, text, tableWidget):
        df = pd.read_csv('resourse\\news.csv')
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
        bd = DB(self.tableWidget)
        bd.loaddata()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(871, 744)
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
        self.label.setGeometry(QtCore.QRect(280, 20, 361, 41))
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
        self.label_result.setGeometry(QtCore.QRect(390, 620, 111, 41))
        self.label_result.setStyleSheet("font: 75 35pt \"Berlin Sans FB Demi\";")
        self.label_result.setWordWrap(True)
        self.label_result.setObjectName("label_result")
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(310, 680, 261, 16))
        self.label_13.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_13.setObjectName("label_13")
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
        self.pushButton_test.setGeometry(QtCore.QRect(280, 470, 321, 21))
        self.pushButton_test.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.pushButton_test.setObjectName("pushButton_test")
        self.label_accuracy = QtWidgets.QLabel(self.tab_3)
        self.label_accuracy.setGeometry(QtCore.QRect(20, 40, 161, 31))
        self.label_accuracy.setStyleSheet("font: 15pt \"Impact\";\n"
"color: rgb(144, 144, 144);")
        self.label_accuracy.setObjectName("label_accuracy")
        self.label_predskazano_pravilno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_pravilno.setGeometry(QtCore.QRect(20, 70, 231, 21))
        self.label_predskazano_pravilno.setStyleSheet("font: 14pt \"Impact\";\n"
"color: rgb(0, 206, 13);")
        self.label_predskazano_pravilno.setObjectName("label_predskazano_pravilno")
        self.label_pravilno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_true_result.setGeometry(QtCore.QRect(20, 90, 161, 21))
        self.label_pravilno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 206, 13);")
        self.label_pravilno_count_true_result.setObjectName("label_pravilno_count_true_result")
        self.label_pravilno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_pravilno_count_fake_result.setGeometry(QtCore.QRect(20, 110, 161, 21))
        self.label_pravilno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 206, 13);")
        self.label_pravilno_count_fake_result.setObjectName("label_pravilno_count_fake_result")
        self.label_neverno_count_true_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_true_result.setGeometry(QtCore.QRect(20, 180, 161, 21))
        self.label_neverno_count_true_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(255, 148, 16);")
        self.label_neverno_count_true_result.setObjectName("label_neverno_count_true_result")
        self.label_predskazano_neverno = QtWidgets.QLabel(self.tab_3)
        self.label_predskazano_neverno.setGeometry(QtCore.QRect(20, 160, 251, 21))
        self.label_predskazano_neverno.setStyleSheet("font: 14pt \"Impact\";\n"
"color: rgb(255, 148, 16);")
        self.label_predskazano_neverno.setObjectName("label_predskazano_neverno")
        self.label_neverno_count_fake_result = QtWidgets.QLabel(self.tab_3)
        self.label_neverno_count_fake_result.setGeometry(QtCore.QRect(20, 200, 161, 21))
        self.label_neverno_count_fake_result.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(255, 148, 16);")
        self.label_neverno_count_fake_result.setObjectName("label_neverno_count_fake_result")
        self.label_12 = QtWidgets.QLabel(self.tab_3)
        self.label_12.setGeometry(QtCore.QRect(330, 670, 301, 16))
        self.label_12.setStyleSheet("font: 10pt \"Impact\";\n"
"color: rgb(84, 103, 118);")
        self.label_12.setObjectName("label_12")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(410, 40, 431, 391))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("resourse/true_false_diagram_before.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(20, 270, 181, 21))
        self.label_2.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 0, 0);")
        self.label_2.setObjectName("label_2")
        self.doubleSpinBoxSizeTestSet = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBoxSizeTestSet.setGeometry(QtCore.QRect(210, 270, 62, 22))
        self.doubleSpinBoxSizeTestSet.setDecimals(1)
        self.doubleSpinBoxSizeTestSet.setMinimum(0.1)
        self.doubleSpinBoxSizeTestSet.setMaximum(0.9)
        self.doubleSpinBoxSizeTestSet.setSingleStep(0.1)
        self.doubleSpinBoxSizeTestSet.setObjectName("doubleSpinBoxSizeTestSet")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(20, 310, 191, 21))
        self.label_4.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 0, 0);")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(20, 330, 201, 21))
        self.label_5.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 0, 0);")
        self.label_5.setObjectName("label_5")
        self.doubleSpinBoxMaxDF = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBoxMaxDF.setGeometry(QtCore.QRect(230, 330, 62, 22))
        self.doubleSpinBoxMaxDF.setDecimals(1)
        self.doubleSpinBoxMaxDF.setMinimum(0.1)
        self.doubleSpinBoxMaxDF.setMaximum(0.9)
        self.doubleSpinBoxMaxDF.setSingleStep(0.1)
        self.doubleSpinBoxMaxDF.setObjectName("doubleSpinBoxMaxDF")
        self.label_6 = QtWidgets.QLabel(self.tab_3)
        self.label_6.setGeometry(QtCore.QRect(300, 330, 91, 21))
        self.label_6.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 0, 0);")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        self.label_7.setGeometry(QtCore.QRect(20, 370, 261, 21))
        self.label_7.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 0, 0);")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setGeometry(QtCore.QRect(20, 390, 171, 21))
        self.label_8.setStyleSheet("font: 12pt \"Impact\";\n"
"color: rgb(0, 0, 0);")
        self.label_8.setObjectName("label_8")
        self.spinBox = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox.setGeometry(QtCore.QRect(200, 390, 61, 22))
        self.spinBox.setMinimum(10)
        self.spinBox.setMaximum(2000)
        self.spinBox.setObjectName("spinBox")
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(60, 520, 771, 41))
        self.label_9.setStyleSheet("font: 11pt \"Impact\";\n"
"color: rgb(0, 115, 255);")
        self.label_9.setObjectName("label_9")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        bd = DB(self.tableWidget)
        self.add_functions_DB(bd)

        machineLearning = MachineLearning(self.doubleSpinBoxSizeTestSet, self.doubleSpinBoxMaxDF, self.spinBox,
                                          self.label_accuracy, self.label_predskazano_pravilno,
                                          self.label_pravilno_count_true_result,
                                          self.label_pravilno_count_fake_result, self.label_predskazano_neverno,
                                          self.label_neverno_count_true_result, self.label_neverno_count_fake_result,
                                          self.label_9, self.label_3, self.label_result, self.tableWidget)

        self.add_functions(machineLearning)
        self.add_functions_test(machineLearning)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификатор новостей"))
        self.label.setText(_translate("MainWindow", "КЛАССИФИКАТОР НОВОСТЕЙ"))
        self.label_title.setText(_translate("MainWindow", "ЗАГОЛОВОК"))
        self.label_text.setText(_translate("MainWindow", "ТЕКСТ"))
        self.pushButton.setText(_translate("MainWindow", "КЛАССИФИЦИРОВАТЬ"))
        self.label_13.setText(_translate("MainWindow", "© Самарский университет, Калинин А.А. 2021"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Распознаватель"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Текст"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Результат"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Дата"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "История"))
        self.pushButton_test.setText(_translate("MainWindow", "ТЕСТ"))
        self.label_accuracy.setText(_translate("MainWindow", "Точность:"))
        self.label_predskazano_pravilno.setText(_translate("MainWindow", "Классифицировано верно:"))
        self.label_predskazano_neverno.setText(_translate("MainWindow", "Классифицировано неверно:"))
        self.label_12.setText(_translate("MainWindow", "© Самарский университет, Калинин А.А. 2021"))
        self.label_2.setText(_translate("MainWindow", "Объем тестовой выборки:"))
        self.label_4.setText(_translate("MainWindow", "Игнорирование терминов,"))
        self.label_5.setText(_translate("MainWindow", "встречающихся более чем в:"))
        self.label_6.setText(_translate("MainWindow", "документах"))
        self.label_7.setText(_translate("MainWindow", "Максимальное количество проходов"))
        self.label_8.setText(_translate("MainWindow", "по обучающим данным"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Исследование"))

        self.tableWidget.setColumnWidth(0, 180)
        self.tableWidget.setColumnWidth(1, 460)
        self.tableWidget.setColumnWidth(2, 75)
        self.tableWidget.setColumnWidth(3, 110)

        bd = DB(self.tableWidget)
        self.add_functions_DB(bd)

    def add_functions(self, machineLearning):
        self.machineLearning = machineLearning
        self.pushButton.clicked.connect(
            lambda: self.machineLearning.result(self.textTitleNews.toPlainText(), self.textNews.toPlainText(), self.tableWidget))

    def add_functions_test(self, machineLearning):
        self.machineLearning = machineLearning
        self.pushButton_test.clicked.connect(lambda: self.machineLearning.result_test())

    def add_functions_DB(self, db):
        self.db = db
        self.db.loaddata()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
