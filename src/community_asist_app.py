import sys
import threading
from util.application_util.gaze_data_receiver import GazeDataReceiver
from util.application_util.Drawer import Drawer

from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QFont
from PyQt5.QtCore import Qt, QRect
from util.application_util.speak import Speak

from random import *

import time


class MyMain(QMainWindow):

    debug_run_speak = True
    debug_fullscreen_mode = True

    xml_dir = "./community_assist_app.xml"
    select_item_sec = 3.0

    def __init__(self):
        super().__init__()


        self.setStyleSheet("background-color: #d1d1d1")

        self.gazeDataReceiver = GazeDataReceiver(self)
        self.drawer = Drawer()
        self.speak = Speak()

        gaze_receive_thread = threading.Thread(target = self.gazeDataReceiver.receive_gaze , name="gaze_receive_thread")
        gaze_receive_thread.daemon = True
        gaze_receive_thread.start()


        self.category_list = []
        self.item_list = []
        self.status = 0

        self.is_start = False
        self.is_speaking = False            # 말하는도중에는 그래픽 업데이트 못하도록
        self.is_checked = False             # 보고있는곳 시작시작 체크했는지

        self.start_time = None

        self.buttons = []
        self.previous_point = None

        # self.gazeDataReceiver.current_point=9
        # self.gazeDataReceiver.correct_point=[2,7,6,1,9]


        self.font = QFont()
        self.font.setPixelSize(50)
        self.parse_xml()                    # 데이터가져오기

        self.setGeometry(450, 300, 1280, 720)
        if self.debug_fullscreen_mode:
            self.setWindowState(Qt.WindowFullScreen)

        self.show()


        ############ debug ##############
        # random_thread = threading.Thread(target=self.create_random)
        # random_thread.daemon = True
        # random_thread.start()

        # update_th = threading.Thread(target=self.update_thread)
        # update_th.daemon = True
        # update_th.start()
        ##################################

    def update_thread(self):
        while(True):
            self.update()
            time.sleep(0.03)


    def resizeEvent(self, QResizeEvent):
        self.window_width = QResizeEvent.size().width()
        self.window_height = QResizeEvent.size().height()


    def create_random(self):
        while True:
            i = randint(1, 4)
            self.gazeDataReceiver.current_point = i

            # self.gazeDataReceiver.correct_point = []
            # self.gazeDataReceiver.correct_point.append(randint(1, 9))
            # self.gazeDataReceiver.correct_point.append(randint(1, 9))

            # print("random point : ", i)
            # print("random correct point : ", self.gazeDataReceiver.correct_point)

            time.sleep(4.05)


    def paintEvent(self, event):

        if self.is_speaking == True:
            return

        if self.is_start == False:
            self.is_start = True
            return

        painter = QPainter(self)

        current_point = self.gazeDataReceiver.current_point
        width = int(self.window_width / 3)
        height = int(self.window_height / 3)

        painter.setFont(self.font)

        for index in range(1, 10):
            x = int((index - 1) % 3)
            y = int((index - 1) / 3)

            # self.drawer.draw_normal_rectangle(painter, x, y, width, height)  # 전체_영역_색칠
            self.drawer.draw_normal_circle(painter, x, y, width, height)

            # 글자 표시
            self.show_text(painter, x, y, width, height, index)


        if self.previous_point is None:
            self.previous_point = current_point
            return

        # start time 측정
        if not self.is_checked:
            self.check_time()

        # 보고있는_포인트가_바뀌면_return
        elif self.previous_point != current_point:
            self.previous_point = current_point
            self.uncheck_time()
            return

        time_diff = round(time.time() - self.start_time, 2)

        #################  Main 카테고리 ##################
        # 3초동안 바라봐서 선택 했을경우.
        if time_diff  > self.select_item_sec:
            # 카테고리 선택
            if self.status == 0:
                # 비어있는 카테고리 선택
                if (current_point > len(self.category_list)):
                    print("카테고리 잘못선택!!")
                    self.status = 0
                    self.uncheck_time()
                    self.update()
                    return

                print("카테고리 선택 : ", self.category_list[current_point-1])
                self.status = current_point
                self.uncheck_time()
                self.update()
                return

            #아이템 선택
            elif self.status > 0:

                if current_point == 9 :
                    print("돌아가기")
                    self.status = 0
                    self.uncheck_time()
                    self.update()
                    return

                if (current_point > len(self.item_list[self.status-1])):
                    print("비어있는 아이템")
                    self.uncheck_time()
                    self.update()
                    return

                print("항목 선택 : ", self.item_list[self.status-1][current_point-1])

                if self.debug_run_speak:
                    self.is_speaking = True
                    self.speak.speak_text(self.item_list[self.status-1][current_point-1])
                    self.is_speaking = False

                self.status = 0
                self.uncheck_time()
                self.update()
                return

        # 3초세는_애니메이션
        else:
            time_diff_rate = round(time_diff / self.select_item_sec, 2)
            x = int((current_point - 1) % 3)
            y = int((current_point - 1) / 3)

            # self.drawer.draw_growing_rectangle(painter, x, y, width, height, time_diff_rate) # 보고있는_포인트_표시
            self.drawer.draw_growing_circle(painter, x, y, width, height, time_diff_rate)

            # 글자 표시
            self.show_text(painter, x, y, width, height, current_point)



    def show_text(self, painter, x, y, width, height, index):
        # Category 선택_창
        if self.status == 0:
            if (index - 1 < len(self.category_list)):               # category 항목 개수만큼만 표시
                self.drawer.draw_text(painter, x, y, width, height, self.category_list[index - 1])
        # 서브레이어_선택_창
        elif self.status > 0:
            if (index - 1 < len(self.item_list[self.status-1])):      # item 항목 개수만큼만 표시
                self.drawer.draw_text(painter, x, y, width, height,
                                  self.item_list[self.status - 1 ][index - 1])
            elif index == 9 :
                self.drawer.draw_text(painter, x, y, width, height, "돌아가기")

    def check_time(self):
        self.is_checked = True
        self.start_time = time.time()

    def uncheck_time(self): self.is_checked = False
    def do_update(self): self.update()

    # xml 가져와서 category_list, item_list 생성
    def parse_xml(self):

        import xml.etree.ElementTree as ET

        doc = ET.parse(self.xml_dir)
        root = doc.getroot()

        for category in root.iter("category"):
            self.category_list.append(category.attrib["name"])

            list = []
            for sub in category.iter("item"):
                list.append(sub.text)
            self.item_list.append(list)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyMain()
    sys.exit(app.exec_())