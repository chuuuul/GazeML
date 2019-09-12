from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap
from PyQt5.QtCore import Qt, QPointF,QObject,pyqtSignal,QRect


class Drawer:

    def draw_normal_rectangle(self,painter, x, y, width, height):
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
        painter.drawRect(QRect(x * width, y * height, width, height))


    def draw_growing_rectangle(self, painter, x, y, width, height, time_diff_rate):
        current_width = int(width * time_diff_rate)
        current_height = int(height * time_diff_rate)
        current_position_x = int((x * width + width * 0.5) - (current_width * 0.5))
        current_position_y = int((y * height + height * 0.5) - (current_height * 0.5))

        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.drawRect(QRect(current_position_x, current_position_y, current_width, current_height))


    def draw_empty_circle(self, painter, x, y, width):
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        painter.drawEllipse(QPointF(x, y), width, width)
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
        painter.drawEllipse(QPointF(x, y), width - 8, width - 8)


    def draw_line_point_to_point(self, painter, x1, y1, x2, y2):
        # painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        painter.setPen(QPen(Qt.black, 4, Qt.DashLine))
        painter.drawLine(x1, y1, x2, y2)
        painter.setPen(QPen())

    def draw_text(self, painter, x, y, width, height, text):
        painter.drawText(QRect(x * width, y * height, width, height), Qt.AlignCenter, text)

    def setColorRed(self, object):
        object.setStyleSheet("background-color:red")


    def setColorWhite(self, object):
        object.setStyleSheet("background-color:white")

