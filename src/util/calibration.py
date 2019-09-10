import queue
import cv2
import numpy as np
import threading
import time


class Calibration:

    def __init__(self):
        pass

    is_face_detect = False

    is_start = False  # type: bool
    is_finish = False
    is_fail = False

    is_full_screen = False

    ###################################### Start Cali #############################################
    Const_Cali_Window_name = 'canvas'

    # 1080p = 1920x1080 / 720p = 1280x720
    Const_Display_X = 1280  # 캘리브레이션 창 넓이
    Const_Display_Y = 720  # 캘리브레이션 창 높이

    # Const_Display_X , self.Const_Display_Y = util.gaze.get_monitor_resolution(debug_monitor_index)

    Const_Cali_Num_X = 4  # 캘리브레이션 포인트 x 갯수
    Const_Cali_Num_Y = 4  # 캘리브레이션 포인트 y 갯수
    Const_Cali_Radius = 20  # 캘리브레이션 포인트 원 크기
    Const_Cali_Goal_Radius = 4  # 캘리브레이션 포인트가 가장 작을 때 원 크기

    Const_Cali_Reduce_Num = 20      # 줄어들 횟수
    Const_Cali_Reduce_Sleep = 0.03  # 한번 줄어드는데 걸리는 시간

    Const_Cali_Move_Num = 20        # 이동할 횟수
    Const_Cali_Move_Sleep = 0.03    # 한번 이동하는데 걸리는 시간

    Const_Cali_Margin_X = 50  # 모니터 모서리에서 떨어질 X 거리
    Const_Cali_Margin_Y = 50  # 모니터 모서리에서 떨어질 Y 거리

    Const_Cali_Cross_Size = 16  # 캘리브레이션 포인트에 십자가 표시 크기

    Const_Grid_Count_X = 3
    Const_Grid_Count_Y = 3

    Cali_Center_Points = []  # 캘리브레이션 좌표

    # 캘리브레이션 값 저장 변수
    left_iris_captured_data = []
    right_iris_captured_data = []

    left_eyeball_captured_data = []
    right_eyeball_captured_data = []

    # 눈 크기 전역 변수

    save_eye_size_x = []
    save_eye_size_y = []

    iris_centre = 0
    eyeball_centre = 0
    eye_size_x = 0
    eye_size_y = 0

    left_iris_centre = 0
    right_iris_centre = 0

    left_eyeball_centre = 0
    right_eyeball_centre = 0

    # 시선 좌표 전역 변수
    left_gaze_coordinate = None
    right_gaze_coordinate = None

    current_point = None
    correct_point = []

    sequence = queue.Queue()
    current_image = None


    def start_cali(self,):

        self.is_start = True

        print("Start Calibration!")

        # 캘리브레이션_순서_설정
        self.make_sequence()

        background = self.init_canvas()
        self.current_image = background.copy()
        self.init_cali()


        # 큐의 순서대로 캘리브레이션 시작
        index = self.sequence.get_nowait()

        tmp_img = background.copy()
        self.draw_circle(tmp_img, self.Cali_Center_Points[index], self.Const_Cali_Radius)
        self.draw_cross(tmp_img, self.Cali_Center_Points[index])
        self.current_image = tmp_img
        time.sleep(1)

        while(True):


            self.resize_figure(background, self.Cali_Center_Points[index], self.Const_Cali_Radius)

            previous_index = index
            index = self.sequence.get_nowait()
            self.move_figure(background, self.Cali_Center_Points[previous_index], self.Cali_Center_Points[index])

            if self.sequence.empty() == True:
                # 마지막_포인트_줄어드는_애니메이션
                self.resize_figure(background, self.Cali_Center_Points[index], self.Const_Cali_Radius)
                break

        self.is_finish = True  # 종료플레그
        print("Complete Calibration!!")


    def resize_figure(self, img, point, radius):

        current_radius = radius
        count = 0


        to_resize_radius = self.Const_Cali_Radius - self.Const_Cali_Goal_Radius       # 총_줄어들어야하는_크기
        # 줄어들크기 / 줄어들_횟수 = 한번에줄을크기
        resize_once_radius = to_resize_radius / self.Const_Cali_Reduce_Num


        while(count != self.Const_Cali_Reduce_Num + 1 ):

            #도화지_초기화
            copy_img = img.copy()

            while (self.is_face_detect == False):
                continue

            # 원_그리기
            self.draw_circle(copy_img, point, current_radius)
            self.draw_cross(copy_img, point)

            #줄어든 횟수 체크 / 다음에_그릴_반지름_계산
            count = count + 1
            current_radius = current_radius - resize_once_radius

            #그림표시
            self.display_canvas(copy_img)

            time.sleep(self.Const_Cali_Reduce_Sleep)


        ############ 캘리브레이션 순간! ############
        self.left_eyeball_captured_data.append(self.left_eyeball_centre)
        self.right_eyeball_captured_data.append(self.right_eyeball_centre)

        self.left_iris_captured_data.append(self.left_iris_centre)
        self.right_iris_captured_data.append(self.right_iris_centre)

        print ("left_iris_captured_data : ",self.left_eyeball_captured_data)
        ##########################################


    def move_figure(self, img, start_point, end_point ):


        count = 0

        to_move_x = (end_point[0] - start_point[0])
        to_move_y = (end_point[1] - start_point[1])

        current_point = (start_point[0],start_point[1])

        # 한번 그릴때마다 이동 할 거리
        move_once_x = to_move_x / (self.Const_Cali_Move_Num)
        move_once_y = to_move_y / (self.Const_Cali_Move_Num)

        while(count != self.Const_Cali_Reduce_Num + 1 ):
            copy_img = img.copy()
            count = count + 1

            self.draw_circle(copy_img, current_point, self.Const_Cali_Radius)
            self.draw_cross(copy_img, current_point)

            current_point = (current_point[0] + move_once_x, current_point[1] + move_once_y)

            self.display_canvas(copy_img)
            time.sleep(self.Const_Cali_Reduce_Sleep)



    def init_cali(self,):

        whole_distance_x = self.Const_Display_X - self.Const_Cali_Margin_X * 2
        whole_distance_y = self.Const_Display_Y - self.Const_Cali_Margin_Y * 2

        unit_distance_x = int(whole_distance_x / (self.Const_Cali_Num_X + (self.Const_Cali_Num_X - 1) - 1))
        unit_distance_y = int(whole_distance_y / (self.Const_Cali_Num_Y - 1))

        print(unit_distance_x)
        print(unit_distance_y)

        for j in range(0,self.Const_Cali_Num_Y):        # 세로
            for i in range(0,self.Const_Cali_Num_X):    # 가로
                y = self.Const_Cali_Margin_Y + unit_distance_y * j
                if (j%2 == 0):      # 4개
                    x = self.Const_Cali_Margin_X + unit_distance_x * 2 * i
                    self.Cali_Center_Points.append( (x, y) )

                if (j%2 == 1):
                    if i == self.Const_Cali_Num_X - 1:
                        continue

                    x = self.Const_Cali_Margin_X + unit_distance_x + unit_distance_x * 2 * i
                    self.Cali_Center_Points.append( (x, y) )

    def init_canvas(self):
        img = np.zeros((self.Const_Display_Y, self.Const_Display_X, 3), np.uint8)
        return img

    def draw_circle(self, img, point, radius):
        cv2.circle(img, ((int)(point[0]), (int)(point[1])), (int)(radius), (0, 0, 255), -1)

    def draw_cross(self, img, point):
        half_size = (int)(self.Const_Cali_Cross_Size / 2)
        cv2.line(img, ((int)(point[0] - half_size), (int)(point[1])),
                       ((int)(point[0] + half_size), (int)(point[1])),
                       (255, 255, 255), 1)
        cv2.line(img, ((int)(point[0]), (int)(point[1] - half_size)),
                       ((int)(point[0]), (int)(point[1] + half_size)),
                       (255, 255, 255), 1)

    def display_canvas(self, img):
        self.current_image=img


    def make_sequence(self):
        self.sequence.put_nowait(0)
        self.sequence.put_nowait(1)
        self.sequence.put_nowait(2)
        self.sequence.put_nowait(3)
        self.sequence.put_nowait(6)
        self.sequence.put_nowait(5)
        self.sequence.put_nowait(4)
        self.sequence.put_nowait(7)
        self.sequence.put_nowait(8)
        self.sequence.put_nowait(9)
        self.sequence.put_nowait(10)
        self.sequence.put_nowait(13)
        self.sequence.put_nowait(12)
        self.sequence.put_nowait(11)















