
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
    show_img = None
    display_size_x = None
    display_size_y = None
    total_size = None

    show_img = cv2.imread('../ballon.jpg', cv2.IMREAD_COLOR)

    def __init__(self, display_size_x, display_size_y ):
        self.display_size_x = display_size_x
        self.display_size_y = display_size_y
        self.total_size = np.sqrt(self.display_size_x * self.display_size_x + self.display_size_y * self.display_size_y)
        print(' test ! total size : ', self.total_size)

        pass



    def calc_error_rate(self,idx):
        error_distance = abs(self.real_coordinates[idx] - self.gaze_coordinates[idx])

        error_rate = error_distance / self.total_size * 100
        self.error_rates.append(error_rate)
        print("인덱스 ", idx," 의 에러율 : ", error_rate)

    def calc_total_error_rate(self,idx):
        sum_rate = 0
        for i in range(0,idx+1):
            sum_rate = sum_rate + self.error_rates[i]

        if sum_rate == 0:
            self.total_rate = 0
        else:
            self.total_rate = sum_rate / idx

        print("전체 에러율 : ", self.total_rate)


    def mouse_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # event

            self.real_coordinates.append( (x,y) )
            self.gaze_coordinates.append( param )
            self.calc_error_rate(self.save_index)
            self.save_index = self.save_index + 1


            # var1 and var2 비교 후 계산

            cv2.circle(self.show_img, (x, y), 3, (255, 0, 0), -1)

        elif event == cv2.EVENT_RBUTTONDBLCLK:
            self.calc_total_error_rate(self.save_index)







cali = Calibration()
perf = Performance(cali.Const_Display_X, cali.Const_Display_Y)

# img = cv2.imread('../ballon.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)  # auto resized
cv2.setWindowProperty('image2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FREERATIO )
cv2.setMouseCallback('image2', perf.mouse_callback,param=[1,2])

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = capture.read()
    perf.show_img = frame

    cv2.imshow('image2', perf.show_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()






