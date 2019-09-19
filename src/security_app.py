import sys
import threading
from util.application_util.gaze_data_receiver import GazeDataReceiver
from util.application_util.Drawer import Drawer
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QRadialGradient
from PyQt5.QtCore import Qt, QPointF, QRect, QPoint

from random import *

import time


class MyMain(QMainWindow):


    debug_fullscreen_mode = True

    def __init__(self):
        super().__init__()

        self.gazeDataReceiver = GazeDataReceiver(self)
        self.drawer = Drawer()

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
        if self.debug_fullscreen_mode:
            self.setWindowState(Qt.WindowFullScreen)

        self.show()

        self.gazeDataReceiver.current_point=9
        self.gazeDataReceiver.correct_point=[2,7,6,1,9]

        ################## Test Code ##################

        # random_thread = threading.Thread(target=self.create_random)
        # random_thread.daemon = True
        # random_thread.start()

        # display_point_thread = threading.Thread(target=self.display_point)
        # display_point_thread.daemon=True
        # display_point_thread.start()

        ###############################################


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
                self.drawer.setColorWhite (self.buttons[self.previous_point - 1])
                self.buttons[self.previous_point - 1].setPixmap(self.off_pixmap)

            self.drawer.setColorRed (self.buttons[self.gazeDataReceiver.current_point - 1])
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
                self.drawer.draw_normal_circle(painter, x, y, width, height)

            # 보고있는_포인트_표시
            else :
                painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                self.drawer.draw_growing_circle(painter, x, y, width, height, 1)
        # 맞은_영역_표시
        previous_history = None
        for i in self.gazeDataReceiver.correct_point:


            x = int((i-1) % 3)
            y = int((i-1) / 3)

            if previous_history is not None:
                previous_x = int((previous_history - 1) % 3)
                previous_y = int((previous_history - 1) / 3)
                self.drawer.draw_line_point_to_point(painter,
                                         previous_x * width + int(width * 0.5),
                                         previous_y * height + int(height * 0.5),
                                         x * width + int(width * 0.5),
                                         y * height + int(height * 0.5))

                #이어주는_선_때문에_묻힌그림_다시그려주기
                self.drawer.draw_empty_circle(painter, previous_x * width + int(width*0.5),
                                       previous_y * height + int(height*0.5), 20)

            previous_history = i

            painter.setBrush(QBrush( Qt.black, Qt.SolidPattern))
            self.drawer.draw_empty_circle(painter, x * width + int(width*0.5), y * height + int(height*0.5), 20)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyMain()
    sys.exit(app.exec_())