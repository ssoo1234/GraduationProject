import sys
import platform
import typing
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

        self.is_UI_opened=False
        self.imgPath=None

    def initUI(self):
        widget=QWidget()

        Hlayout=QHBoxLayout()
        widget.setLayout(Hlayout)

        self.imageLabel=CView()
        self.imageLabel.setMinimumSize(515,515)
        self.imageLabel.setMaximumSize(515,515)
        Hlayout.addWidget(self.imageLabel)

        self.Vlayout=QVBoxLayout()
        Hlayout.addLayout(self.Vlayout)

        btnSelect=QPushButton('select image',self)
        btnSelect.resize(btnSelect.sizeHint())
        btnSelect.clicked.connect(self.btn_FileLoad)
        self.Vlayout.addWidget(btnSelect)

        self.modeLayout=QVBoxLayout()
        self.Vlayout.addLayout(self.modeLayout)

        self.clsLayout=QVBoxLayout()
        self.Vlayout.addLayout(self.clsLayout)

        self.clsLayout.addStretch(1)

        exitAction=QAction(QIcon('./images/exit.png'),'Exit',self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('exit')
        exitAction.triggered.connect(qApp.quit)

        menuBar=self.menuBar()
        if platform.system()=='Darwin':
            menuBar.setNativeMenuBar(False)
        filemenu=menuBar.addMenu('&File')
        filemenu.addAction(exitAction)

        self.statusBar().showMessage('Ready')

        widget.setLayout(self.Vlayout)

        self.setCentralWidget(widget)
        
        self.setGeometry(300,300,700,600) # x, y, w, h
        self.setWindowTitle('title')
        # self.show()

    def modeUI(self):
        self.is_UI_opened=True

        self.modeLayout.setStretch(0,0)

        self.btnDrawMode=QRadioButton('Draw Mode',self)
        self.btnDrawMode.setChecked(True)
        self.btnDrawMode.toggled.connect(self.get_radio)
        self.modeLayout.addWidget(self.btnDrawMode)
        
        self.btnSelectMode=QRadioButton('Select Mode',self)
        self.btnSelectMode.toggled.connect(self.get_radio)
        self.modeLayout.addWidget(self.btnSelectMode)

        btnReset=QPushButton('Reset Mask',self)
        btnReset.resize(btnReset.sizeHint())
        btnReset.clicked.connect(self.clearMask)
        self.modeLayout.addWidget(btnReset)

        self.btnInpainting=QPushButton('Inpainting',self)
        self.btnInpainting.resize(self.btnInpainting.sizeHint())
        self.btnInpainting.clicked.connect(self.inpainting)
        self.modeLayout.addWidget(self.btnInpainting)

        btnSave=QPushButton('Save image',self)
        btnSave.resize(btnSave.sizeHint())
        btnSave.clicked.connect(self.saveImage)
        self.modeLayout.addWidget(btnSave)

        self.modeLayout.addStretch(1)

    def clearMask(self):
        self.imageLabel.clearMask()
        print("clear")
        if self.imgPath!=None:
            self.imageLabel.setBackground(self.imgPath)

    def saveImage(self):
        self.imageLabel.saveImage()

    def inpainting(self):
        self.imageLabel.inpainting()

    def clsUI(self,clss):
        # if not self.clsLayout.isEmpty():
            # self.clsLayout.re

        if len(clss)>10:
            clss=clss[:10]

        self.btnclss=[]
        for cls in clss:
            self.btnclss.append(QPushButton(cls,self))
        for btn in self.btnclss:
            btn.resize(btn.sizeHint())
            btn.clicked.connect(self.cls_Clicked)
            self.clsLayout.addWidget(btn)

    def cls_Clicked(self):
        btn=self.sender()
        # print(btn.text())
        self.imageLabel.select_cls(btn.text())

    def get_radio(self,checked):
        if checked:
            if self.btnDrawMode.isChecked():
                self.imageLabel.setmode(False)
                print('draw Mode')
            else:
                self.imageLabel.setmode(True)
                print("select Mode")

    def btn_FileLoad(self):
        filter='Image files (*.jpg *.jpeg *.png)'
        fname=QFileDialog.getOpenFileName(self,filter=filter)

        if fname[0]:
            self.imgPath=fname[0]

            self.imageLabel.setBackground(self.imgPath)
            if not self.is_UI_opened:
                self.modeUI()
            self.cls_names=self.imageLabel.get_clss()
            print(self.cls_names)
            self.clsUI(self.cls_names)

        else:
            # print('Image not selected')
            QMessageBox.about(self,'Warning','Image not selected')

class CView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.imgPath=None
        self.__mode=False
        self.aa=asdf()
        self.maskimage=maskImage()
        self.resImage=None

        self.scene=QGraphicsScene()
        self.setScene(self.scene)

        self.items=[]

        self.start=QPointF()
        self.end=QPointF()

        self.maskColor=QColor(0,255,0,128)

        # self.setRenderHint(QPainter.high)
        # self.scene.clear()

    def setBackground(self,imgPath):
        self.imgPath=imgPath
        self.pixmap=QPixmap(imgPath).scaled(512,512)
        self.graphicsPixmapItem = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.graphicsPixmapItem)

        self.aa.predict(imgPath)
        
        return
    
    def setmode(self,cond):
        self.__mode=cond
        return
    
    def get_clss(self):
        return self.aa.get_cls()
    
    def select_cls(self,cls):
        masks=self.aa.select_clss(cls)
        for mask in masks:
            self.maskimage.add_mask(mask)
            self.drawMask(mask)
        # self.maskimage.show()

    def drawMask(self,mask):
        color=QColor(0,255,0,64)
        brush=QBrush(color)
        pen = QPen(color)

        for x in range(mask.shape[0]):
            for y in range (mask.shape[1]):
                if mask[x][y]!=0:
                    ellipse = QGraphicsRectItem(0, 0, 1, 1)
                    ellipse.setBrush(brush)
                    ellipse.setPen(pen)
                    ellipse.setPos(y, x)  # 항목 위치 설정
                    self.scene.addItem(ellipse)
        # pass

    def mousePressEvent(self,e):
        if self.__mode:
            if e.button()==Qt.LeftButton:
                self.start = e.pos()
                self.end = e.pos()
                
            if self.aa.get_pred():
                mask=self.aa.get_seg(self.start.x(),self.start.y())
                if mask is not None:
                    self.maskimage.add_mask(mask)
                    self.drawMask(mask)
                # print(self.start.x(),self.start.y())

                # demo(self.imgPath,self.maskimage.get_mask())

        return
    
    def inpainting(self):
        self.resImage=demo(self.imgPath,self.maskimage.get_mask())
        
        self.saveImage()
        # cv2.imwrite('/Users/hwangseho/Documents/GitHub/pyqt/images/res/peoples.jpg', self.resImage)
        pixmap=QPixmap('/Users/hwangseho/Documents/GitHub/pyqt/images/res/peoples.jpg')
        graphicsPixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(graphicsPixmapItem)

    def clearMask(self):
        self.scene.clear()
        self.maskimage=maskImage()
    
    def saveImage(self):
        cv2.imwrite('/Users/hwangseho/Documents/GitHub/pyqt/images/res/peoples.jpg', self.resImage)

from ultralytics import YOLO
class asdf():
    def __init__(self):
        self.model=YOLO('./model/yolov8s-seg.pt')
        self.__is_predicted=False

        self.res=None
        self.clss=None

    def predict(self,imgPath):
        self.res=self.model(imgPath,imgsz=512)
        
        for r in self.res:
            clss=r.boxes.cls.numpy().astype('int16')
            self.clss=np.unique(clss)
        
        self.__is_predicted=True
        return
    
    def select_clss(self,cls):
        names=self.model.names
        masks=[]
        for r in self.res:
            for box,mask in zip(r.boxes,r.masks):
                # print(int(box.cls[0]))
                if names[int(box.cls[0])]==cls:
                    masks.append(mask.data[0].numpy())

        return masks
    
    def get_seg(self,y,x):
        for r in self.res:
            for mask in r.masks:
                m=mask.data[0].numpy()

                if m[x][y]!=0:
                    # print(m[x][y])
                    return m
                
        return None

    def get_cls(self):
        names=self.model.names
        
        clsNames=[names[int(c)] for c in self.clss]
        return clsNames
    
    def get_pred(self):
        return self.__is_predicted

from PIL import Image
class maskImage():
    def __init__(self):
        self.maskArr=np.zeros((512,512,),dtype=np.uint8)

    def add_mask(self,mask):
        self.maskArr=np.where(mask,mask,self.maskArr)
        # self.show()

    def get_mask(self):
        return self.maskArr

    def show(self):
        mask_img=Image.fromarray(np.uint8(255*self.maskArr))
        mask_img.show()

import cv2
import os
import importlib
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from utils.option import args

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def demo(img_path,mask):
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load('/Users/hwangseho/Documents/GitHub/pyqt/G0000000.pt', map_location='cpu'))
    model.eval()

    filename = img_path
    orig_img = cv2.resize(cv2.imread(filename, cv2.IMREAD_COLOR), (512, 512))
    img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
    mask = mask

    # print('[**] inpainting ... ')
    with torch.no_grad():
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

        comp_np = postprocess(comp_tensor[0])

        # cv2.imwrite('/Users/hwangseho/Documents/GitHub/pyqt/images/res/0.jpg', comp_np)

    return comp_np

if __name__=='__main__':
    app=QApplication(sys.argv)
    ex=MyApp()
    ex.show()
    sys.exit(app.exec_())