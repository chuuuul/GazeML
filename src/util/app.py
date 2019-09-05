import sys
import threading
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QPointF,QObject,pyqtSignal


class Emitter(QObject):

    emitSignal = pyqtSignal()

class MyMain(QMainWindow):
    def __init__(self):
        super().__init__()

        self.emitter = Emitter()
        self.emitter.emitSignal.connect(self.do_update)

        self.statusbar = self.statusBar()
        self.is_start = False
        self.mx = -10
        self.my = -10
        self.once = False

        print(self.hasMouseTracking())
        self.setMouseTracking(True)   # True 면, mouse button 안눌러도 , mouse move event 추적함.
        print(self.hasMouseTracking())

        self.setGeometry(300, 200, 1280, 720)
        self.show()

    def do_update(self):
        print("emiittttt")
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
        painter = QPainter(self)
        painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
        painter.drawEllipse(QPointF( self.mx, self.my), 10, 10)
        if self.once == False:
            self.once = True
            th = threading.Timer(1,self.send_emit)
            th.start()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = MyMain()
    sys.exit(app.exec_())