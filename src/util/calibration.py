
import queue
import cv2
import numpy as np
import threading


class Calibration:

    tmp_var = 2

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
    Const_Cali_Radius = 30  # 캘리브레이션 포인트 원 크기
    Const_Cali_Resize_Radius = 7  # 캘리브레이션 포인트가 가장 작을 때 원 크기

    Const_Cali_Unit_Time = 60  # 캘리브레이션 한 번 표현 소요 시간 (쓰레드 반복 시간, 프레임) # 1/60 초마다 실행 # 50 일 때 문제발생
    Const_Cali_Move_Duration = 0.2  # 캘리브레이션 원 이동 횟수       # 이동 할 때 (Unit_Time * Move_Duration)만큼 소요
    Const_Cali_Capture_Duration = 0.2  # 캘리브레이션 원 줄어드는 횟수    # 줄어들 때 (Unit_Time * Move_Duration)만큼 소요

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

    sequence = queue.Queue()


    def start_cali(self,):

        self.is_start = True

        print("Start Calibration!")
        # 큐에 캘리브레이션 순서 인덱스를 찾례로 넣는다
        for i in range(0, self.Const_Cali_Num_X * self.Const_Cali_Num_Y):
            self.sequence.put_nowait(i)

        img = self.init_canvas()
        self.init_cali()

        # 큐의 순서대로 캘리브레이션 시작
        index = self.sequence.get_nowait()

        self.resize_figure(img, self.Cali_Center_Points[index], self.Const_Cali_Radius, self.Const_Cali_Capture_Duration)

        # 문제점 : waitkey로 키 입력 받으려고 변수 저장 후 비교해서 다르면 waitkey를 다시 하게 되는데 그 순간 먹통(쓰레드랑관련?)
        cv2.waitKey(0)
        print("pressed key!! Exit calibration!")
        self.is_finish = True
        self.is_fail = True
        self.close_window(self.Const_Cali_Window_name)
        return

    def move_figure(self, img, start_point, end_point, current_point, duration, count=0):

        while (self.is_face_detect == False):
            continue

        copy_img = img.copy()

        to_move_x = (end_point[0] - start_point[0])
        to_move_y = (end_point[1] - start_point[1])

        move_once_x = to_move_x / (duration * self.Const_Cali_Unit_Time)
        move_once_y = to_move_y / (duration * self.Const_Cali_Unit_Time)

        updated_current_point = (current_point[0] + move_once_x, current_point[1] + move_once_y)

        self.draw_circle(copy_img, current_point, self.Const_Cali_Radius)
        self.draw_cross(copy_img, current_point)

        self.display_canvas(self.Const_Cali_Window_name, copy_img)
        count = count + 1

        if (count == (duration * self.Const_Cali_Unit_Time)):
            self.resize_figure(img, end_point, self.Const_Cali_Radius, self.Const_Cali_Capture_Duration, )
            return

        th = threading.Timer(1 / self.Const_Cali_Unit_Time, self.move_figure,
                             [img, start_point, end_point, updated_current_point, self.Const_Cali_Move_Duration,
                              count])
        th.daemon = True
        th.start()

    def resize_figure(self, img, point, current_radius, duration, count=0):

        copy_img = img.copy()

        to_resize_radius = self.Const_Cali_Radius - self.Const_Cali_Resize_Radius
        resize_once_radius = to_resize_radius / (duration * self.Const_Cali_Unit_Time)

        updated_current_radius = current_radius - resize_once_radius
        self.draw_circle(copy_img, point, updated_current_radius)
        self.draw_cross(copy_img, point)

        self.display_canvas(self.Const_Cali_Window_name, copy_img)
        count = count + 1



        while (self.is_face_detect == False):
            continue

        # 캘리브레이션 순간!
        if (count == (duration * self.Const_Cali_Unit_Time)):

            ##########################################
            # to-do : 눈의 좌표 저장
            # idea : 개선점? : 캘리브레이션 중간에 값 저장해서 보정하는건 어떤가?

            self.left_eyeball_captured_data.append(self.left_eyeball_centre)
            self.right_eyeball_captured_data.append(self.right_eyeball_centre)

            self.left_iris_captured_data.append(self.left_iris_centre)
            self.right_iris_captured_data.append(self.right_iris_centre)

            print ("left_iris_captured_data : ",self.left_eyeball_captured_data)

            ##########################################


            # 다음 캘리브레이션 경로가 없다면 창 종료
            if (self.sequence.empty() == True):
                self.is_finish = True  # 종료플레그
                print("Complete Calibration!!")
                self.close_window(self.Const_Cali_Window_name)
                return


            # 큐에 다음 캘리브레이션 포인트가 있다면 원을 이동하여 캘리브레이션 작업
            index = self.sequence.get_nowait()
            self.move_figure(img, point, self.Cali_Center_Points[index], point, self.Const_Cali_Move_Duration)
            return

        th = threading.Timer(1 / self.Const_Cali_Unit_Time, self.resize_figure,
                             [img, point, updated_current_radius, duration, count])
        th.daemon = True
        th.start()

    def init_cali(self,):
        # 캘리브레이션 포인인
        cali_unit_distance_x = 0
        cali_unit_distance_y = 0

        # 캘리브레이션이 가로나 세로에 1개라면 계산식에서 나누기 에러 발생
        # 실제로 캘리브레이션을 1개만 하는 경우는 없고 디버깅 용도이므로 자세하게 코딩하지 않았다.
        if self.Const_Cali_Num_X != 1:
            cali_unit_distance_x = (self.Const_Display_X - self.Const_Cali_Margin_X * 2) / (self.Const_Cali_Num_X - 1)

        if self.Const_Cali_Num_Y != 1:
            cali_unit_distance_y = (self.Const_Display_Y - self.Const_Cali_Margin_Y * 2) / (self.Const_Cali_Num_Y - 1)

        for y in range(0, self.Const_Cali_Num_Y):
            for x in range(0, self.Const_Cali_Num_X):
                self.Cali_Center_Points.append(((int)(self.Const_Cali_Margin_X + cali_unit_distance_x * x)
                                           , (int)(self.Const_Cali_Margin_Y + cali_unit_distance_y * y)))

    def init_canvas(self,):
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

    def display_canvas(self, canvas_name, img):
        # Display On Canvas
        if self.is_full_screen :
            cv2.namedWindow(canvas_name, cv2.WINDOW_NORMAL)  # auto resized
            cv2.setWindowProperty(canvas_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(canvas_name, img)
        None

    def close_window(self, canvas_name):
        # cv2.destroyWindow(canvas_name)
        cv2.destroyAllWindows()