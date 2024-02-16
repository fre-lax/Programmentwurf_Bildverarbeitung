import sys
import importlib
import os
import json
import copy
import traceback
from pprint import *

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import matplotlib.pyplot as plt

import cv2
import numpy as np

class ImageWidget(QLabel):
    pixelValueEvt = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.image_org=None
        self.scale=1
        self.drag=False
        self.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setMinimumSize(100,100)
        self.pos=QPointF(0,0)
        self.setStyleSheet("QLabel { background-color : #606060;  padding: 0px 0px 0px 0px; }")

    def wheelEvent(self, a0: QWheelEvent) -> None:
        if(a0.angleDelta().y()>0): 
            self.scale=self.scale*1.1
            if(self.scale>10): self.scale=10
        else:
            self.scale=self.scale*0.9
            if(self.scale<1): self.scale=1

        if(self.scale>1):
            mousePosViewRel=self.mousePercentPosition(a0.position())
            
            mousePosImageAbs=self.mousePixelPosition(a0.position(),self.viewRect)

            newWidth=int(self.image_org.shape[1]/self.scale)
            newHeight=int(self.image_org.shape[0]/self.scale)

            posX=int(mousePosImageAbs.x()-mousePosViewRel.x()*newWidth)
            posY=int(mousePosImageAbs.y()-mousePosViewRel.y()*newHeight)

            self.viewRect=QRect(posX,posY,newWidth,newHeight)
        else:
            self.viewRect=QRect(0,0,self.image_org.shape[1],self.image_org.shape[0])

        self.updateView()

        return super().wheelEvent(a0)
    
    def resizeEvent(self, event):
        
        if(self.image_org is None): return
        height,width=self.geometry().height(),self.geometry().width()
        self.scale_percent = min(width/self.image_org.shape[1],height / self.image_org.shape[0])
        self.control_width = int(self.image_org.shape[1] * self.scale_percent)
        self.control_height = int(self.image_org.shape[0] * self.scale_percent)
        
        self.updateView()
        self.setPixmap(self.pixmap_from_cv_image( self.image_mod))

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if(ev.buttons()== Qt.MouseButton.LeftButton and hasattr(self,"viewRect")):
            self.drag=True
            self.mousePos=ev.localPos()
            self.viewRectPos=copy.deepcopy(self.viewRect)
        else:
            self.drag=False

        return super().mousePressEvent(ev)
    
    def mouseDoubleClickEvent(self, a0: QMouseEvent) -> None:
        if(hasattr(self,"viewRect")):
            self.pos=self.mousePixelPosition(a0.localPos(),self.viewRect)
            self.updatePixelValue()
        return super().mouseDoubleClickEvent(a0)
    
    def updatePixelValue(self):
        try:
            val=self.image_org[int(self.pos.y()),int(self.pos.x())]
        except:
            val=""
        self.pixelValueEvt.emit("("+str(self.pos.y())+","+str(self.pos.x())+") "+str(val))

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if (ev.buttons() == Qt.MouseButton.LeftButton) and self.drag:
            self.viewRect=QRectF(self.viewRectPos)
            self.viewRect.translate(-self.mousePixelPosition(ev.localPos(),self.viewRectPos)+self.mousePixelPosition(self.mousePos,self.viewRectPos))
            self.updateView()
        return super().mouseMoveEvent(ev)
    
    def mousePixelPosition(self,pos,viewRect):
        posP=self.mousePercentPosition(pos)
        return QPoint(int(viewRect.left()+posP.x()*viewRect.width()),int(viewRect.top()+posP.y()*viewRect.height()))

    def mousePercentPosition(self,pos):
        x=pos.x()/self.control_width
        y=pos.y()/self.control_height
        return QPointF(x,y)

    def updateView(self):
        if(self.viewRect.top()+self.viewRect.height()>self.image_org.shape[0]):
            self.viewRect.moveTop(self.image_org.shape[0]-self.viewRect.height())
        if(self.viewRect.left()+self.viewRect.width()>self.image_org.shape[1]):
            self.viewRect.moveLeft(self.image_org.shape[1]-self.viewRect.width())
        if(self.viewRect.top()<0): self.viewRect.moveTop(0)
        if(self.viewRect.left()<0): self.viewRect.moveLeft(0)

        h,w=self.geometry().height(),self.geometry().width()     

        scale=self.scale*self.scale_percent
        viewRectTmp=QRectF(self.viewRect)
        viewRectTmp.setWidth(int(w/scale))
        viewRectTmp.setHeight(int(h/scale))

        self.image_mod=self.image_org[int(viewRectTmp.top()):int(viewRectTmp.top()+viewRectTmp.height()+1),int(viewRectTmp.left()):int(viewRectTmp.left()+viewRectTmp.width()+1)]

        self.image_mod = cv2.resize(self.image_mod, None, None, self.scale_percent*self.scale, self.scale_percent*self.scale, cv2.INTER_AREA)
        self.setPixmap(self.pixmap_from_cv_image( self.image_mod))

    def pixmap_from_cv_image(self,cv_image):
        height, width, *_ = cv_image.shape
        if(len(cv_image.shape)==3):
            qImg = QImage(cv_image.data, width, height, 3 * width, QImage.Format.Format_RGB888).rgbSwapped()
        else:
            qImg = QImage(cv_image.data, width, height, width, QImage.Format.Format_Grayscale8).rgbSwapped()
        return QPixmap(qImg)
    
    def setData(self,image_data):
        if(len(image_data.shape)==1):
            fig=plt.figure()
            plt.plot(image_data)
            fig.tight_layout()
            plt.grid(True)
            plt.xlim(0.0, len(image_data)-1)

            fig.canvas.draw()
            image_data = np.array(fig.canvas.renderer._renderer)
            image_data=image_data[:,:,0:3]
            plt.close()

        sameFormat=False
        if(self.image_org is not None):
            if(self.image_org.shape[1]==image_data.shape[1] and self.image_org.shape[0]==image_data.shape[0]):
                sameFormat=True
        if(image_data.dtype == np.float64) or (image_data.dtype == np.float32):
            self.image_org=(image_data*255.99).astype(np.uint8)
        else:
            self.image_org=image_data
        self.updatePixelValue()

        if(not sameFormat):
            rect=self.geometry()
            self.scale=1
            self.scale_percent = min((rect.width())/self.image_org.shape[1],(rect.height()) / self.image_org.shape[0])
            self.control_width = (self.image_org.shape[1] * self.scale_percent)
            self.control_height = (self.image_org.shape[0] * self.scale_percent)
            self.viewRect=QRectF(0,0,self.image_org.shape[1],self.image_org.shape[0])

        self.updateView()
        self.setPixmap(self.pixmap_from_cv_image( self.image_mod))

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        try:
            self.config=json.load(open('ImageProcessingGUI.json'))   
        except:
            self.config = {'folder' : os.getcwd(), 'WinRect' : [100,100,400,400], 'scriptIndex' : 0 , 'val1' : 0,'val2' : 0,}
        self.setGeometry(self.config['WinRect'][0],self.config['WinRect'][1],self.config['WinRect'][2],self.config['WinRect'][3])

        self.setWindowTitle("Image Processing GUI v.2.1 - "+self.config['folder'])

        main_layout = QVBoxLayout()
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self.source_image_data = None

        select_image_button = QPushButton('Folder')
        select_image_button.clicked.connect(self.choose_source_folder)
        reload_button = QPushButton('Reload')
        reload_button.clicked.connect(self.reload)
        saveImage_button = QPushButton('Save')
        saveImage_button.clicked.connect(self.saveImage)

        for btn in [select_image_button,reload_button,saveImage_button]:
            btn.setFixedHeight(50)
            btn.setFixedWidth(120)

        self.sld2 = QSlider(Qt.Orientation.Horizontal, self)
        self.sld2L = QLabel()
        self.sld1 = QSlider(Qt.Orientation.Horizontal, self)
        self.sld1L = QLabel()
        for start_val, slider, label in zip([self.config["val1"],self.config["val2"]], [self.sld1,self.sld2], [self.sld1L,self.sld2L]):
            #slider.sliderReleased.connect(self.update)
            slider.setRange(0, 255)
            slider.setPageStep(1)
            slider.setValue(start_val)
            label.setText(str(start_val))
            #slider.setFixedWidth(300)
            slider.valueChanged.connect(self.update)

        self.textbox = QLineEdit(self)
        self.textbox.setReadOnly(True)
        self.textbox.setFixedWidth(300)

        self.comboScripts = QComboBox(self)
        for file in os.listdir("."):
            if (file.endswith(".py") and file!=os.path.basename(__file__)) and not "ImageProcessingGUI" in file:
                f = open(file, "r").read()
                if(f.__contains__("def run(")):
                    try:
                        module = importlib.import_module(file.rsplit('.', 1)[0])
                        if(hasattr(module, "run")):
                            self.comboScripts.addItem(file)
                    except Exception as err:
                        print("Script "+file+" not loaded: "+str(err))

                    
        if(self.config["scriptIndex"]<0 or self.config["scriptIndex"]>=self.comboScripts.count()): self.config["scriptIndex"]=0
        self.comboScripts.setCurrentIndex(self.config["scriptIndex"])
        self.comboScripts.currentTextChanged.connect(self.update)

        list_layout =QDockWidget('Image List',self)
        list_layout2 =QDockWidget('Result List',self)
        self.imageList = QListWidget(self)
        self.resultList = QListWidget(self)

        list_layout.setWidget(self.imageList)
        list_layout2.setWidget(self.resultList)

        top_bar_layout.addWidget(select_image_button)
        top_bar_layout.addWidget(reload_button)
        top_bar_layout.addWidget(saveImage_button)
        top_bar_layout.addWidget(self.comboScripts)
        top_bar_layout.addWidget(self.sld1)
        top_bar_layout.addWidget(self.sld1L)
        top_bar_layout.addWidget(self.sld2)
        top_bar_layout.addWidget(self.sld2L)
        top_bar_layout.addWidget(self.textbox)
        
        self.image_view = ImageWidget()

        self.image_view.pixelValueEvt.connect(self.changed)

        self.imageResultList=[]
        
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,list_layout)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,list_layout2)
        list_layout.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        list_layout2.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetMovable)

        main_layout.addLayout(top_bar_layout)
        main_layout.addWidget(self.image_view)#,alignment=Qt.AlignmentFlag.AlignCenter)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        if not 'imageIndex' in self.config: self.config["imageIndex"]=[0,0]
        self.load_folder()
        self.imageList.setCurrentRow(self.config["imageIndex"][0])
        self.load_Image()       
        self.resultList.setCurrentRow(self.config["imageIndex"][1])

    def saveImage(self):
        name=self.resultList.currentItem().text()
        result=QFileDialog.getSaveFileName(self,"Speichern",self.config['folder']+"\\"+name+".jpg","Bilder (*.png *.jpg)")
        if(result[0]!=""):
            img=self.resultListData[self.config["imageIndex"][1]]
            cv2.imwrite(result[0],self.resultListData[self.config["imageIndex"][1]])

    def reload(self):
        if(hasattr(self,"msg")): self.msg.close()
        self.load_Image()

    def update(self):
        self.sld1L.setText(str(self.sld1.value()))
        self.sld2L.setText(str(self.sld2.value()))
        self.save()
        self.load_Image()

    def changed(self,text):
        self.textbox.setText(text)

    def save(self):
        rect=self.geometry()
        self.config['WinRect']=[rect.x(),rect.y(),rect.width(),rect.height()]
        self.config["scriptIndex"]=self.comboScripts.currentIndex()
        self.config["val1"]=self.sld1.value()
        self.config["val2"]=self.sld2.value()
        json.dump(self.config, open('ImageProcessingGUI.json', 'w'),default=lambda x: x.__dict__)

    def choose_source_folder(self):
        result=QFileDialog.getExistingDirectory(directory=self.config['folder'])
        if(result):
            self.config["folder"]=result
            self.setWindowTitle("Image Processing GUI - "+self.config['folder'])

            self.save()
            self.load_folder()
            self.load_Image()

    def load_folder(self):
        try: self.imageList.currentItemChanged.disconnect()
        except: pass
        self.imageList.clear()
        for file in os.listdir(self.config["folder"]):
            if file.lower().endswith((".jpg",".png",".tiff",".tif")):
                self.imageList.addItem(QListWidgetItem(file))
        self.imageList.setCurrentRow(1)
        self.imageList.currentItemChanged.connect(self.image_changed)

    def image_changed(self,cur,prev):
        cur=self.imageList.currentRow()
        self.load_Image()
        self.config["imageIndex"][0]=cur

    def result_changed(self,cur,prev):
        cur=self.resultList.currentRow()
        self.image_view.setData(self.resultListData[cur])
        self.config["imageIndex"][1]=cur

    def load_Image(self):
        try: self.resultList.currentItemChanged.disconnect()
        except: pass

        self.resultList.clear()
        self.resultListData=[]
        self.process()
        try:
            self.image_view.setData(self.resultListData[self.config["imageIndex"][1]])
            self.resultList.setCurrentRow(self.config["imageIndex"][1])
        except Exception as e: pass
        self.resultList.currentItemChanged.connect(self.result_changed)

    def addResultImage(self,name,data):
        self.resultList.addItem(QListWidgetItem(name))
        self.resultListData.append(data)

    def closeEvent(self, event):
        if(hasattr(self,"msg")): self.msg.close()
        self.save()
  
    def process(self):
        if(self.imageList.currentItem() is None): return
        name=self.imageList.currentItem().text()
        name=os.path.join( self.config["folder"],name)
        image=cv2.imread(name)
        self.addResultImage("Base",image)
        try:
            script=str(self.comboScripts.currentText())

            if(script!=""):
                del script.rsplit('.', 1)[0]
                module = __import__(script.rsplit('.', 1)[0])
                importlib.reload(module)
                moduleFct = getattr(module, "run")
                result=[]
                moduleFct(image,result,[self.sld1.value(),self.sld2.value(),])
                for ele in result:
                    self.addResultImage(ele["name"],ele["data"])
        except Exception as err:
            if(hasattr(self,"msg")): self.msg.close()
            self.msg = QMessageBox(self)
            self.msg.setAttribute(Qt.WA_ShowWithoutActivating, True)
            self.msg.setWindowFlags(Qt.WindowStaysOnTopHint)
            self.msg.setWindowTitle("Fehler")
            self.msg.setText("Error {0}".format(traceback.format_exc()))
            self.msg.setModal(False)
            self.msg.show()
            self.msg.move(self.geometry().width() - self.msg.width(), self.geometry().height() - self.msg.height())
            traceback.print_exception(type(err), err, err.__traceback__)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()