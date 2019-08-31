
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
    last_path = []

    show_img = cv2.imread('../ballon.jpg', cv2.IMREAD_COLOR)

    def __init__(self, display_size_x, display_size_y ):
        self.display_size_x = display_size_x
        self.display_size_y = display_size_y
        self.total_size = np.sqrt(self.display_size_x * self.display_size_x + self.display_size_y * self.display_size_y)
        print(' test ! total size : ', self.total_size)

        pass



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


    def mouse_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # event

            self.real_coordinates.append( (x,y) )
            self.gaze_coordinates.append( param )
            self.calc_error_rate(self.save_index)
            self.save_index = self.save_index + 1
            self.write_history(x,y)


            # var1 and var2 비교 후 계산


        elif event == cv2.EVENT_MBUTTONDOWN:
            self.calc_total_error_rate(self.save_index)

    def write_history(self,x,y):
        self.last_path = (x, y)

    def draw_mark(self):
        if self.save_index != 0:
            cv2.circle(self.show_img, self.last_path, 7, (255, 0, 0), -1)




########### 추가 ##################
import time  # time 라이브러리

###################################
perf = Performance(1280, 720)

cam = cv2.VideoCapture(-1)  # 카메라 생성
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 윈도우 생성 및 사이즈 변경
cv2.namedWindow('CAM_Window')
cv2.setMouseCallback('CAM_Window', perf.mouse_callback,param=[10,10])



prevTime = 0  # 이전 시간을 저장할 변수
while (True):

    ret, perf.show_img = cam.read()

    curTime = time.time()

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)


    # 프레임 수를 문자열에 저장
    str = "FPS : %0.1f" % fps

    # 표시
    cv2.putText(perf.show_img, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


    if perf.save_index != 0 :
        perf.draw_mark()
    # 얻어온 이미지 윈도우에 표시
    cv2.imshow('CAM_Window', perf.show_img)

    # 10ms 동안 키입력 대기
    if cv2.waitKey(10) >= 0:
        break;

# 윈도우 종려
cam.release()
cv2.destroyWindow('CAM_Window')


