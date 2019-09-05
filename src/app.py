import sys
import threading
from util.gaze_data_receiver import GazeDataReceiver
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QPointF,QObject,pyqtSignal


class Emitter(QObject):

    emitSignal = pyqtSignal()

class MyMain(QMainWindow):
    def __init__(self):
        super().__init__()

        self.gazeDataReceiver = GazeDataReceiver(self)


        gaze_receive_thread = threading.Thread(target = self.gazeDataReceiver.receive_gaze , name="gaze_receive_thread")
        gaze_receive_thread.daemon = True
        gaze_receive_thread.start()

        self.emitter = Emitter()
        self.emitter.emitSignal.connect(self.do_update)

        self.statusbar = self.statusBar()
        self.is_start = False
        self.mx = -10
        self.my = -10
        self.once = False
        self.buttons = []

        self.make_buttons()

        self.setMouseTracking(True)   # True 면, mouse button 안눌러도 , mouse move event 추적함
        self.setGeometry(300, 200, 1280, 720)
        self.show()

    def setColorRed(self,object):
        object.setStyleSheet("background-color:red")

    # def button_init(self,object):
    #     object.setAutoDefault(True)


    def make_buttons(self):
        for i in range(0,9):
            btn = QPushButton('Button'+str(i), self)
            # btn.setGeometry(i*100,i*100,130,30)
            self.buttons.append(btn)
            # btn.setGeometry()




    def resizeEvent(self, QResizeEvent):
        # print (QResizeEvent)
        self.window_width = QResizeEvent.size().width()
        self.window_height = QResizeEvent.size().height()

        width = int(self.window_width / 3)
        height = int(self.window_height / 3)

        for i in range(0,9):
            row = int( i / 3)
            column = int (i % 3)
            self.buttons[i].setGeometry(column*width,row*height,width,height)
        #
        # QPushButton.resize
        # self.buttons[0].set = self.window_width
        # self.buttons[0]. = self.window_height



    def do_update(self):
        # print("emiittttt")
        self.update()

    def send_emit(self):
        self.emitter.emitSignal.emit()

    def mouseMoveEvent(self, event):
        txt = "Mouse 위치 ; x={0},y={1}, global={2},{3}".format(event.x(), event.y(), event.globalX(), event.globalY())
        self.mx = event.x()
        self.my = event.y()


        self.statusbar.showMessage(txt)
        self.update()
        # print(event.globalX())

    def paintEvent(self, event):
        if self.is_start == False:
            self.is_start = True
            return
        if self.gazeDataReceiver.gaze_x is None or self.gazeDataReceiver.gaze_y is None :
            return

        painter = QPainter(self)
        painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        painter.drawEllipse(QPointF(  self.gazeDataReceiver.gaze_x, self.gazeDataReceiver.gaze_y), 10, 10)

        import numpy as np
        if self.once == False:
            self.once = True
            th = threading.Timer(1,self.send_emit)
            th.start()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = MyMain()
    sys.exit(app.exec_())