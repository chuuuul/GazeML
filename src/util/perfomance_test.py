
import threading
import numpy as np
import cv2
import time
import numpy as np

from util.calibration import Calibration

class Performance:

    real_coordinates = []           # 실제로 입력한 좌표
    gaze_coordinates = []           # 보고있는 좌표
    error_rates = []                # 에러율
    save_index = 0

    display_size_x = None
    display_size_y = None
    total_size = None
    last_real_coordinate = []
    last_gaze_coordinate = []

    is_set_callback = False


    def __init__(self, display_size_x, display_size_y ):
        self.display_size_x = display_size_x
        self.display_size_y = display_size_y
        self.total_size = np.sqrt(self.display_size_x * self.display_size_x + self.display_size_y * self.display_size_y)


    def calc_error_rate(self,idx):
        error_distance_x = self.real_coordinates[idx][0] - self.gaze_coordinates[idx][0]
        error_distance_y = self.real_coordinates[idx][1] - self.gaze_coordinates[idx][1]

        error_distance = np.sqrt(error_distance_x * error_distance_x + error_distance_y * error_distance_y)

        error_rate = error_distance / self.total_size * 100
        self.error_rates.append(error_rate)
        print("인덱스 ", idx," 의 에러율 : ", error_rate)


    def calc_total_error_rate(self,idx):
        sum_rate = 0
        for i in range(0,idx):
            sum_rate = sum_rate + self.error_rates[i]

        if sum_rate == 0:
            self.total_rate = 0
        else:
            self.total_rate = sum_rate / idx

        print("전체 에러율 : ", self.total_rate)


    # param 은 캘리브레이션 객체
    def mouse_callback(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # event
            # print("Left Button Click !!!")
            tmp_gaze_coordinate = (param.left_gaze_coordinate + param.right_gaze_coordinate) / 2
            gaze_coordinate = (int (tmp_gaze_coordinate[0]) , int (tmp_gaze_coordinate[1]))

            self.real_coordinates.append( (x,y) )
            self.gaze_coordinates.append( gaze_coordinate )
            self.calc_error_rate(self.save_index)
            self.save_index = self.save_index + 1
            self.write_last_real_coordinate((x,y))
            self.write_last_gaze_coordinate(gaze_coordinate)


        elif event == cv2.EVENT_MBUTTONDOWN:
            if self.save_index == 0:
                return

            self.calc_total_error_rate(self.save_index)


    def write_last_real_coordinate(self, coordinate):
        self.last_real_coordinate = coordinate


    def write_last_gaze_coordinate(self, coordinate):
        self.last_gaze_coordinate = coordinate


    def draw_real_coordinate_mark(self,img):
        if self.save_index != 0:
            cv2.circle(img, self.last_real_coordinate, 7, (255, 0, 0), -1)


    def draw_gaze_coordinate_mark(self,img):
        if self.save_index != 0:
            cv2.circle(img, self.last_gaze_coordinate, 7, (0, 0, 255), -1)

