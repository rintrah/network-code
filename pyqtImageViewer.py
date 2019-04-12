#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:11:38 2019

@author: enrique
"""

import sys 
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import  QPixmap, QPainter, QPen 
from PyQt5 import QtCore
import tifffile
import numpy as np 
import qimage2ndarray
import csv

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title  = 'PyQt5 Image'
        self.left   = 10
        self.top    = 10 
        self.width  = 640
        self.height = 480
        self.initUI()
        self.points = []
        self.scene = QGraphicsScene()
        self.view  = QGraphicsView()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Create widget.
        self.label   = QLabel(self)
        pathToImage  = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image File (*.tif)")
        image        = tifffile.imread(pathToImage[0])
        image        = (image - image.min())/(image.max() - image.min())
        image        = np.uint8(image*255)
        self.pixmap  = QPixmap(qimage2ndarray.array2qimage(image))
        self.image   = image
        self.label.setPixmap(self.pixmap)
        self.installEventFilter(self)
        self.resize(self.pixmap.width(),self.pixmap.height())
        self.show()
        
    def paintEvent(self, paint_event):
        if self.points:
            painter = QPainter(self)
            pen = QPen()
            pen.setWidth(20)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing, True)
            for pos in self.points:
                painter.drawPoint(pos[0], pos[1])
            painter.end()
    
    def mouseReleaseEvent(self, QMouseEvent):
        #print('(', QMouseEvent.x(), ' , ', QMouseEvent.y(), ')')
        x = QMouseEvent.x()
        y = QMouseEvent.y()
        self.points.append((x, y))
        self.update()
        
        
    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply    = QMessageBox.question(self, 'Message', \
                      quit_msg, QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            file = open('/home/enrique/points.csv', 'w')
            writer = csv.writer(file, dialect='excel')
            for pos in self.points:
                writer.writerow([pos[0], pos[1], 1])
            event.accept()
        else:
            event.ignore()
    """        
    def eventFilter(self, source, event):
        if (source is self and event.type() == QtCore.QEvent.Resize):
            image = QPixmap(qimage2ndarray.array2qimage(self.image))
            self.pixmap = image.scaled(self.size())
            self.label.setPixmap(self.pixmap)
            #self.label.setPixmap(self.pixmap.scaled(self.size()))
            self.label.resize(self.pixmap.width(), self.pixmap.height())
        return super(App, self).eventFilter(source, event)
     """   
if __name__ == '__main__':
    app  = QApplication(sys.argv)
    ex   = App()
    sys.exit(app.exec_())