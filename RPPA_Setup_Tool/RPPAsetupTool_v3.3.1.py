# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RPPAsetupUI3.0.4.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os
import CreateArray6, CreateArray9, CreateArray12
import Array_to_6Plates, Array_to_9Plates, Array_to_12Plates
import rename

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 420)
        MainWindow.setMinimumSize(QtCore.QSize(600, 420))
        MainWindow.setMaximumSize(QtCore.QSize(600, 420))
        font = QtGui.QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 10, 580, 400))
        self.tabWidget.setMinimumSize(QtCore.QSize(580, 400))
        self.tabWidget.setMaximumSize(QtCore.QSize(580, 400))
        palette = QtGui.QPalette()
        self.tabWidget.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("background-image:url(:/whiteBackground/white-background.jpg)")
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setObjectName("tabWidget")
        self.Array = QtWidgets.QWidget()
        self.Array.setAutoFillBackground(False)
        self.Array.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.Array.setObjectName("Array")
        self.plate9_1 = QtWidgets.QRadioButton(self.Array)
        self.plate9_1.setGeometry(QtCore.QRect(230, 105, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setItalic(True)
        self.plate9_1.setFont(font)
        self.plate9_1.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.plate9_1.setIconSize(QtCore.QSize(11, 11))
        self.plate9_1.setObjectName("plate9_1")
        self.plate9_1.toggled.connect(lambda: self.selectedRadio1(2)) ##############
        
        self.plate12_1 = QtWidgets.QRadioButton(self.Array)
        self.plate12_1.setGeometry(QtCore.QRect(230, 150, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setItalic(True)
        self.plate12_1.setFont(font)
        self.plate12_1.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.plate12_1.setIconSize(QtCore.QSize(11, 11))
        self.plate12_1.setObjectName("plate12_1")
        self.plate12_1.toggled.connect(lambda: self.selectedRadio1(3))
        
        self.plate6_1 = QtWidgets.QRadioButton(self.Array)
        self.plate6_1.setGeometry(QtCore.QRect(230, 60, 131, 41))
        self.plate6_1.setMaximumSize(QtCore.QSize(211, 41))
        self.plate6_1.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setItalic(True)
        self.plate6_1.setFont(font)
        self.plate6_1.setAutoFillBackground(False)
        self.plate6_1.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.plate6_1.setIconSize(QtCore.QSize(11, 11))
        self.plate6_1.setObjectName("plate6_1")
        self.plate6_1.toggled.connect(lambda: self.selectedRadio1(1))
        
        self.groupBox_1 = QtWidgets.QGroupBox(self.Array)
        self.groupBox_1.setGeometry(QtCore.QRect(200, 40, 441, 170))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_1.setFont(font)
        self.groupBox_1.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.groupBox_1.setStyleSheet("border-radius: 40px; \n"
"background-color: rgba(0, 0, 0, 40);")
        self.groupBox_1.setTitle("")
        self.groupBox_1.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.groupBox_1.setCheckable(False)
        self.groupBox_1.setObjectName("groupBox_1")
        self.widget_3 = QtWidgets.QWidget(self.Array)
        self.widget_3.setGeometry(QtCore.QRect(-10, 0, 721, 391))
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.widget_3.setFont(font)
        self.widget_3.setAutoFillBackground(False)
        self.widget_3.setStyleSheet("background-image: url(:/backgrounds/backgound600x400.png);")
        self.widget_3.setObjectName("widget_3")
        self.MaroonSlip = QtWidgets.QWidget(self.widget_3)
        self.MaroonSlip.setGeometry(QtCore.QRect(40, 240, 161, 61))
        self.MaroonSlip.setStyleSheet("border-radius: 10px;\n"
"background-image: url(:/backgrounds/maroon.jpg);")
        self.MaroonSlip.setObjectName("MaroonSlip")
        self.label_1 = QtWidgets.QLabel(self.MaroonSlip)
        self.label_1.setGeometry(QtCore.QRect(29, -3, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(10)
        self.label_1.setFont(font)
        self.label_1.setStyleSheet("background-image: rgba(0, 0, 0, 0);border-radius: 0px;")
        self.label_1.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_1.setObjectName("label_1")
        self.graphicsView_1 = QtWidgets.QGraphicsView(self.widget_3)
        self.graphicsView_1.setGeometry(QtCore.QRect(40, 260, 521, 111))
        self.graphicsView_1.setStyleSheet("border-width:1px; border-style: solid;border-style: solid;border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(158,158,158, 255), stop:1 rgba(202, 202, 202, 255));\n"
"background-image: url(:/backgrounds/cool-background.png);\n"
"border-top-color: rgba(158, 158, 158,255);border-top-width:10px")
        self.graphicsView_1.setObjectName("graphicsView_1")
        self.bnCreateArray = QtWidgets.QPushButton(self.widget_3)
        self.bnCreateArray.setGeometry(QtCore.QRect(70, 140, 91, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.bnCreateArray.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.bnCreateArray.setFont(font)
#         self.bnCreateArray.setStyleSheet("border-radius: 15px;\n"
# "background-image: url(:/buttons/button-W3.png);")
        self.bnCreateArray.setStyleSheet("QPushButton"
                             "{"
                             "background-image:url(:/buttons/button-W3.png);border-radius: 15px"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-image:url(:/buttons/button-B2.png);border-radius: 15px"
                             "}"
                             )           
        self.bnCreateArray.setObjectName("bnCreateArray")
        self.bnCreateArray.clicked.connect(self.runTab_1) ##############
        
        self.bnBrowseList = QtWidgets.QPushButton(self.widget_3)
        self.bnBrowseList.setGeometry(QtCore.QRect(70, 70, 91, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.bnBrowseList.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.bnBrowseList.setFont(font)
#         self.bnBrowseList.setStyleSheet("border-radius: 15px;\n"
# "background-image: url(:/buttons/button-W3.png);")
        self.bnBrowseList.setStyleSheet("QPushButton"
                             "{"
                             "background-image:url(:/buttons/button-W3.png);border-radius: 15px"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-image:url(:/buttons/button-B2.png);border-radius: 15px"
                             "}"
                             )   
        self.bnBrowseList.setObjectName("bnBrowseList")
        self.bnBrowseList.clicked.connect(self.inputTab_1) ##############
        
        self.widget_3.raise_()
        self.groupBox_1.raise_()
        self.plate9_1.raise_()
        self.plate12_1.raise_()
        self.plate6_1.raise_()
        self.tabWidget.addTab(self.Array, "")
        self.Plates = QtWidgets.QWidget()
        self.Plates.setStyleSheet("")
        self.Plates.setObjectName("Plates")
        self.bnArraySheet = QtWidgets.QPushButton(self.Plates)
        self.bnArraySheet.setGeometry(QtCore.QRect(60, 70, 91, 31))
        palette = QtGui.QPalette()
        self.bnArraySheet.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        self.bnArraySheet.setFont(font)
#         self.bnArraySheet.setStyleSheet("border-radius: 15px;\n"
# "background-image: url(:/buttons/button-W3.png);")
        self.bnArraySheet.setStyleSheet("QPushButton"
                     "{"
                     "background-image:url(:/buttons/button-W3.png);border-radius: 15px"
                     "}"
                     "QPushButton::pressed"
                     "{"
                     "background-image:url(:/buttons/button-B2.png);border-radius: 15px"
                     "}"
                     )     
        self.bnArraySheet.setObjectName("bnArraySheet")
        self.bnArraySheet.clicked.connect(self.inputTab_2) ##############
        
        self.bnCreatePlates = QtWidgets.QPushButton(self.Plates)
        self.bnCreatePlates.setGeometry(QtCore.QRect(60, 140, 91, 31))
        palette = QtGui.QPalette()
        self.bnCreatePlates.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        self.bnCreatePlates.setFont(font)
#         self.bnCreatePlates.setStyleSheet("border-radius: 15px;\n"
# "background-image: url(:/buttons/button-W3.png);")
        self.bnCreatePlates.setStyleSheet("QPushButton"
                             "{"
                             "background-image:url(:/buttons/button-W3.png);border-radius: 15px"
                             "}"
                             "QPushButton::pressed"
                             "{"
                             "background-image:url(:/buttons/button-B2.png);border-radius: 15px"
                             "}"
                             )         
        
        self.bnCreatePlates.setObjectName("bnCreatePlates")
        self.bnCreatePlates.clicked.connect(self.runTab_2) ##############
        
        self.groupBox_2 = QtWidgets.QGroupBox(self.Plates)
        self.groupBox_2.setGeometry(QtCore.QRect(200, 40, 461, 170))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAutoFillBackground(False)
        self.groupBox_2.setStyleSheet("border-radius: 40px; border-width:0px; border-style: solid; border-style: solid; border-color: rgb(171,171,171);\n"
"background-color: rgba(0, 0, 0, 40);")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.plate6_2 = QtWidgets.QRadioButton(self.Plates)
        self.plate6_2.setGeometry(QtCore.QRect(230, 60, 131, 41))
        self.plate6_2.setMaximumSize(QtCore.QSize(211, 41))
        self.plate6_2.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setItalic(True)
        self.plate6_2.setFont(font)
        self.plate6_2.setAutoFillBackground(False)
        self.plate6_2.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.plate6_2.setIconSize(QtCore.QSize(11, 11))
        self.plate6_2.setObjectName("plate6_2")
        self.plate6_2.toggled.connect(lambda: self.selectedRadio2(1))
        
        self.plate12_2 = QtWidgets.QRadioButton(self.Plates)
        self.plate12_2.setGeometry(QtCore.QRect(230, 150, 131, 41)) ##############
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setItalic(True)
        self.plate12_2.setFont(font)
        self.plate12_2.setAutoFillBackground(False)
        self.plate12_2.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.plate12_2.setIconSize(QtCore.QSize(11, 11))
        self.plate12_2.setObjectName("plate12_2")
        self.plate12_2.toggled.connect(lambda: self.selectedRadio2(3))
        
        self.plate9_2 = QtWidgets.QRadioButton(self.Plates)
        self.plate9_2.setGeometry(QtCore.QRect(230, 105, 131, 41)) ##############
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(12)
        font.setItalic(True)
        self.plate9_2.setFont(font)
        self.plate9_2.setAutoFillBackground(False)
        self.plate9_2.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        self.plate9_2.setIconSize(QtCore.QSize(11, 11))
        self.plate9_2.setObjectName("plate9_2")
        self.plate9_2.toggled.connect(lambda: self.selectedRadio2(2)) ##############
        
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.Plates)
        self.graphicsView_2.setGeometry(QtCore.QRect(30, 260, 521, 111))
        self.graphicsView_2.setStyleSheet("border-width:1px; border-style: solid;border-style: solid;border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(158,158,158, 255), stop:1 rgba(202, 202, 202, 255));\n"
"background-image: url(:/backgrounds/cool-background.png);\n"
"border-top-color: rgba(158, 158, 158,255);border-top-width:10px")
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.widget_2 = QtWidgets.QWidget(self.Plates)
        self.widget_2.setGeometry(QtCore.QRect(-10, 0, 591, 381))
        self.widget_2.setAutoFillBackground(False)
        self.widget_2.setStyleSheet("background-image: url(:/backgrounds/backgound600x400.png);")
        self.widget_2.setObjectName("widget_2")
        self.OrangeSlip = QtWidgets.QWidget(self.widget_2)
        self.OrangeSlip.setGeometry(QtCore.QRect(40, 240, 161, 61))
        self.OrangeSlip.setStyleSheet("border-radius: 10px;\n"
"background-image: url(:/backgrounds/orange3.jpg);")
        self.OrangeSlip.setObjectName("OrangeSlip")
        self.label_2 = QtWidgets.QLabel(self.OrangeSlip)
        self.label_2.setGeometry(QtCore.QRect(29, -3, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgba(0, 0, 0, 0);border-radius: 0px;")
        self.label_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_2.setObjectName("label_2")
        self.widget_2.raise_()
        self.groupBox_2.raise_()
        self.bnArraySheet.raise_()
        self.bnCreatePlates.raise_()
        self.plate6_2.raise_()
        self.plate12_2.raise_()
        self.plate9_2.raise_()
        self.graphicsView_2.raise_()
        self.tabWidget.addTab(self.Plates, "")
        self.Rename = QtWidgets.QWidget()
        self.Rename.setStyleSheet("background-image:url(:/newPrefix/backgound600x400.png)")
        self.Rename.setObjectName("Rename")
        self.widget_4 = QtWidgets.QWidget(self.Rename)
        self.widget_4.setGeometry(QtCore.QRect(-30, 0, 881, 501))
        self.widget_4.setAutoFillBackground(False)
        self.widget_4.setStyleSheet("background-image: url(:/backgrounds/cool-background.png);")
        self.widget_4.setObjectName("widget_4")
        self.bnStart = QtWidgets.QPushButton(self.widget_4)
        self.bnStart.setGeometry(QtCore.QRect(490, 200, 81, 31))
        self.bnBrowse = QtWidgets.QPushButton(self.widget_4)
        self.bnBrowse.setGeometry(QtCore.QRect(490, 155, 81, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        
        self.bnBrowse.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Helvetica Rounded")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.bnBrowse.setFont(font)
        self.bnBrowse.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.bnBrowse.setStyleSheet("border-radius: 15px")
        self.bnBrowse.setFlat(False)
        self.bnBrowse.setObjectName("bnBrowse")
        self.bnBrowse.clicked.connect(self.inputTab_3) ##############
        
        self.bnStart.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Helvetica Rounded")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.bnStart.setFont(font)
        self.bnStart.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.bnStart.setStyleSheet("border-radius: 15px")
        self.bnStart.setFlat(False)
        self.bnStart.setObjectName("bnStart")
        self.bnStart.clicked.connect(self.runTab_3) ##############        
        
        
        self.lineEdit = QtWidgets.QLineEdit(self.widget_4)
        self.lineEdit.setGeometry(QtCore.QRect(340, 100, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.lineEdit.setFont(font)
        self.lineEdit.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lineEdit.setAutoFillBackground(False)
        self.lineEdit.setStyleSheet("border-radius: 15px; border-width:2px; border-style: solid;background-color: rgba(255, 255, 255,0);\n"
"border-color:rgba(255, 125, 11, 200);\n"
"font: 12pt \"Helvetica\";")
        self.lineEdit.setText("")
        self.lineEdit.setMaxLength(14)
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setObjectName("lineEdit")
        
        
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.widget_4)
        self.graphicsView_3.setGeometry(QtCore.QRect(30, 260, 575, 111))
        self.graphicsView_3.setStyleSheet("border-radius: 0px;border-width:0px; border-style: solid;border-style: solid;border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(158,158,158, 255), stop:1 rgba(202, 202, 202, 255));\n"
"background-image: url(:/backgrounds/cool-background.png);\n"
"border-top-color: rgba(158, 158, 158,255);border-top-width:5px")
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.renameTitle = QtWidgets.QGraphicsView(self.widget_4)
        self.renameTitle.setGeometry(QtCore.QRect(100, -10, 521, 91))
        self.renameTitle.setStyleSheet("background-image: url(:/backgrounds/Rename_002.PNG);border-radius: 15px; border-width:2px; border-style: solid;\n"
"border-color: rgba(255, 255, 255, 50);\n"
"border-bottom-color: rgba(158, 158, 158,155);border-bottom-width:3px")
        self.renameTitle.setObjectName("renameTitle")
        self.tabWidget.addTab(self.Rename, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionClose.triggered.connect(self.close_application)
        
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.triggered.connect(self.myAbout)
        
        self.menuFile.addAction(self.actionClose)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RPPA Setup Tool"))
        self.plate9_1.setText(_translate("MainWindow", "9 Plates"))
        self.plate12_1.setText(_translate("MainWindow", "12 Plates"))
        self.plate6_1.setText(_translate("MainWindow", "6 Plates"))
        self.label_1.setText(_translate("MainWindow", "Command Display"))
        self.bnCreateArray.setText(_translate("MainWindow", "Generate"))
        self.bnBrowseList.setText(_translate("MainWindow", "Input"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Array), _translate("MainWindow", "Array"))
        self.bnArraySheet.setText(_translate("MainWindow", "Input"))
        self.bnCreatePlates.setText(_translate("MainWindow", "Generate"))
        self.plate6_2.setText(_translate("MainWindow", "6 Plates"))
        self.plate12_2.setText(_translate("MainWindow", "12 Plates"))
        self.plate9_2.setText(_translate("MainWindow", "9 Plates"))
        self.label_2.setText(_translate("MainWindow", "Command Display"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Plates), _translate("MainWindow", "Plates + Layouts"))
        self.bnBrowse.setText(_translate("MainWindow", "Browse"))
        self.bnStart.setText(_translate("MainWindow", "Start"))
        self.lineEdit.setToolTip(_translate("MainWindow", "Type in the experiment name here"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Rename), _translate("MainWindow", "Rename"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionAbout.setText(_translate("MainWindow", "About"))

    def inputTab_1(self):
        (fileName, fileType) = QtWidgets.QFileDialog.getOpenFileName(self,
                                                         'Select PIs List File',
                                                         '.',
                                                         '*.xlsx')
        self.pathTab_1 = fileName
        self.text1 = os.path.basename(str(self.pathTab_1))
        self.viewText1()
    
    def inputTab_2(self):
        (fileName, fileType) = QtWidgets.QFileDialog.getOpenFileName(self,
                                                         'Select Array File',
                                                         '.',
                                                         '*.xlsx')
        self.pathTab_2 = fileName
        self.text2 = os.path.basename(str(self.pathTab_2))
        self.viewText2()

    def inputTab_3(self):
        (fileName, fileType) = QtWidgets.QFileDialog.getOpenFileName(self,
                                                         'Select Array File',
                                                         '.',
                                                         '*.txt')
        self.pathTab_3 = fileName
        self.text3 = os.path.basename(str(self.pathTab_3))
        self.viewText3()
        
    def selectedRadio1(self, n):
        if n == 1:
            self.text1 = "Plate-6 configuration selected"
            self.select1 = 1
            self.viewText1()
        elif n == 2:
            self.text1 = "Plate-9 configuration selected"
            self.select1 = 2
            self.viewText1()
        elif n == 3:
            self.text1 = "Plate-12 configuration selected"
            self.select1 = 3
            self.viewText1()            

    def selectedRadio2(self, n):
        if n == 1:
            self.text2 = "Plate-6 configuration selected"
            self.select2 = 1
            self.viewText2()
        elif n == 2:
            self.text2 = "Plate-9 configuration selected"
            self.select2 = 2
            self.viewText2()
        elif n == 3:
            self.text2 = "Plate-12 configuration selected"
            self.select2 = 3
            self.viewText2()   

    def viewText1(self):
        scene = QtWidgets.QGraphicsScene(self)
        scene.addText(self.text1)
        self.graphicsView_1.setScene(scene)
        
    def viewText2(self):
        scene = QtWidgets.QGraphicsScene(self)
        scene.addText(self.text2)
        self.graphicsView_2.setScene(scene)

    def viewText3(self):
        scene = QtWidgets.QGraphicsScene(self)
        scene.addText(self.text3)
        self.graphicsView_3.setScene(scene)
        
    def saveFile(self):
        saveFileName, ext = QtWidgets.QFileDialog.getSaveFileName(self, 'Save as')
        
        if saveFileName:
            file = open(saveFileName + '.txt', 'w')
            text = str(self.textTab_1)
            file.write(text)
            file.close()
            
#            DF = self.textTab_1
#            writer = pd.ExcelWriter(saveFileName + '.xlsx', engine='xlsxwriter')
#            DF.to_excel(writer, 'samples')
#            writer.save()
            
            self.text1 = "Array was saved as\n" + os.path.basename(str(saveFileName))  + ".txt"
            self.viewText1()

    def runTab_1(self):        
#        self.text = "PI list loaded from\n" + os.path.basename(str(self.pathTab_1))
#        self.viewText()
        
        if self.select1 == 1:
            self.textTab_1 = CreateArray6.main(str(self.pathTab_1)) ## this does not return as a dataframe
        elif self.select1 == 2:
            self.textTab_1 = CreateArray9.main(str(self.pathTab_1))
        elif self.select1 == 3:
            self.textTab_1 = CreateArray12.main(str(self.pathTab_1))
        else:
            self.text1 = "Please choose a Plate option"
            self.viewText1()
            return
            
        self.text1 = "Array Created.\nPlease save."
        self.viewText1()
        
        self.saveFile()

    def runTab_2(self):        
        self.text2 = "processing..."
        self.viewText2()       
        
        if self.select2 == 1:
            Array_to_6Plates.main(str(self.pathTab_2)) ## this does not return as a dataframe
        elif self.select2 == 2:
            Array_to_9Plates.main(str(self.pathTab_2))
        elif self.select2 == 3:
            Array_to_12Plates.main(str(self.pathTab_2))
        else:
            self.text2 = "Please choose a Plate option"
            self.viewText2()
            return
            
        self.text2 = "Plates and Layouts generated"
        self.viewText2()
        
        # self.saveFile()

    def runTab_3(self):        
      
        if self.lineEdit.text() == '':
            self.text3 = "Experiment is name missing"
            self.viewText3()
        # elif self.pathTab_3 == '':
        #     self.text3 = "Staining list is missing"
        #     self.viewText3()            
        else:
            rename.main(self.lineEdit.text(), str(self.pathTab_3)) 
            self.text3 = "Rename Complete!"
            self.viewText3()

    def myAbout(self):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("\nRPPA SetupTool 3.3.0\nCreated by Dimuthu Perera\nCoarfa Lab")
        msgBox.setWindowTitle("About RPPAtool")
        msgBox.exec_()

    def close_application(self):
        choice = QtWidgets.QMessageBox.question(self, 'Exit',
                                            "This will close the program. Are you sure?",
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()
        else:
            pass        

import resources_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

