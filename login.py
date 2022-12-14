# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(361, 307)
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
        self.label_error_login_or_pass.setGeometry(QtCore.QRect(120, 180, 131, 16))
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

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_in.setText(_translate("Form", "??????????"))
        self.pushButton_reg.setText(_translate("Form", "????????????????????????????????????"))
        self.lineEdit_login.setPlaceholderText(_translate("Form", "??????????"))
        self.lineEdit_pass.setPlaceholderText(_translate("Form", "????????????"))
        self.label_name_system.setText(_translate("Form", "?????????????????????????? ????????????????"))
        self.label_have_not_acc.setText(_translate("Form", "?????? ?????????????????"))
        self.label_authorization.setText(_translate("Form", "??????????????????????"))
        self.label_error_login_or_pass.setText(_translate("Form", "???????????????? ????????????"))
        self.label_enter_login_and_pass.setText(_translate("Form", "?????????????? ?????????? ?? ????????????"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
