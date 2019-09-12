import sys
import threading
from util.application_util.gaze_data_receiver import GazeDataReceiver
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QPointF, QRect

from random import *

import time


class MyMain(QMainWindow):
    def __init__(self):
        super().__init__()

        self.gazeDataReceiver = GazeDataReceiver(self)

        gaze_receive_thread = threading.Thread(target = self.gazeDataReceiver.receive_gaze , name="gaze_receive_thread")
        gaze_receive_thread.daemon = True
        gaze_receive_thread.start()

        self.statusbar = self.statusBar()
        self.is_start = False
        self.mx = -10
        self.my = -10

        self.buttons = []
        self.previous_point = None

        self.setMouseTracking(True)   # True 면, mouse button 안눌러도 , mouse move event 추적함
        self.setGeometry(300, 200, 1280, 720)
        self.show()

        self.gazeDataReceiver.current_point=9
        self.gazeDataReceiver.correct_point=[2,7,6,1,9]

        # random_thread = threading.Thread(target=self.create_random)
        # random_thread.daemon = True
        # random_thread.start()

        # display_point_thread = threading.Thread(target=self.display_point)
        # display_point_thread.daemon=True
        # display_point_thread.start()



    def setColorRed(self,object):
        object.setStyleSheet("background-color:red")

    def setColorWhite(self,object):
        object.setStyleSheet("background-color:white")

    def make_buttons(self):
        for i in range(0,9):
            # btn = QPushButton('Button'+str(i), self)
            # self.buttons.append(btn)
            btn = QLabel('Button'+str(i), self)

            self.buttons.append(btn)
    def resizeEvent(self, QResizeEvent):
        self.window_width = QResizeEvent.size().width()
        self.window_height = QResizeEvent.size().height()


    def do_update(self):
        self.update()


    def mouseMoveEvent(self, event):
        txt = "Mouse 위치 ; x={0},y={1}, global={2},{3}".format(event.x(), event.y(), event.globalX(), event.globalY())
        self.mx = event.x()
        self.my = event.y()
        self.statusbar.showMessage(txt)


        self.update()

    def create_random(self):
        while True:
            i = randint(1, 9)

            self.gazeDataReceiver.correct_point = []
            self.gazeDataReceiver.correct_point.append(randint(1, 9))
            self.gazeDataReceiver.correct_point.append(randint(1, 9))

            print("random point : ", i)
            print("random correct point : ", self.gazeDataReceiver.correct_point)
            self.gazeDataReceiver.current_point = i
            self.do_update()
            time.sleep(1)


    def display_point(self):
        while self.gazeDataReceiver.current_point is None:
            continue

        while True:
            if self.previous_point is not None:
                self.setColorWhite (self.buttons[self.previous_point - 1])
                self.buttons[self.previous_point - 1].setPixmap(self.off_pixmap)

            self.setColorRed (self.buttons[self.gazeDataReceiver.current_point - 1])
            self.buttons[self.gazeDataReceiver.current_point - 1].setPixmap(self.on_pixmap)
            self.previous_point = self.gazeDataReceiver.current_point



    def paintEvent(self, event):
        if self.is_start == False:
            self.is_start = True
            return

        width = int(self.window_width / 3)
        height = int(self.window_height / 3)

        painter = QPainter(self)
        for i in range(1, 10):

            x = int((i-1) % 3)
            y = int((i-1) / 3 )

            # 전체_영역_색칠
            if i != self.gazeDataReceiver.current_point:
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                painter.drawRect(QRect(x * width, y * height, width, height))

            # 보고있는_포인트_표시
            else :
                painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                painter.drawRect(QRect(x * width, y * height, width, height))

        # if self.gazeDataReceiver.correct_point !=
        # print (self.gazeDataReceiver.correct_point)

        # 맞은_영역_표시
        previous_history = None
        for i in self.gazeDataReceiver.correct_point:


            x = int((i-1) % 3)
            y = int((i-1) / 3)

            if previous_history is not None:
                previous_x = int((previous_history - 1) % 3)
                previous_y = int((previous_history - 1) / 3)
                self.draw_line_point_to_point(painter,
                                         previous_x * width + int(width * 0.5),
                                         previous_y * height + int(height * 0.5),
                                         x * width + int(width * 0.5),
                                         y * height + int(height * 0.5))

                #이어주는_선_때문에_묻힌그림_다시그려주기
                self.draw_empty_circle(painter, previous_x * width + int(width*0.5),
                                       previous_y * height + int(height*0.5), 20)

            previous_history = i



            painter.setBrush(QBrush( Qt.black, Qt.SolidPattern))
            self.draw_empty_circle(painter, x * width + int(width*0.5), y * height + int(height*0.5), 20)




    def draw_empty_circle(self,painter,x,y,width):
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        painter.drawEllipse(QPointF(x, y), width, width)
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
        painter.drawEllipse(QPointF(x, y), width-8, width-8)

    def draw_line_point_to_point(self,painter,x1,y1,x2,y2):

        # painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        painter.setPen(QPen(Qt.black, 4, Qt.DashLine))
        painter.drawLine(x1,y1,x2,y2)
        painter.setPen(QPen())



    #     #  Gaze값을 표현 할 때
    #     if self.gazeDataReceiver.gaze_x is None or self.gazeDataReceiver.gaze_y is None :
    #         return
    #     painter = QPainter(self)
    #     painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
    #     painter.drawEllipse(QPointF( self.mx, self.my), 10, 10)
    #
    #
    #
    #     if self.once == False:
    #         self.once = True
    #         th = threading.Timer(1,self.send_emit)
    #         th.start()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = MyMain()
    sys.exit(app.exec_())