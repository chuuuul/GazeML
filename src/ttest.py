# Tensorflow 확 인코드
# import tensorflow as tf
# hello = tf.constant('hello tensol')
# sess = tf.Session()
# print(sess.run(hello))


import threading
import numpy as np
import cv2
import time

def pr():
    print("print_Thread!!")




def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # event
        # var1.append(실제찍은값)
        # var2.append(보고있는값)
        #
        # var1 and var2 비교 후
        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)

        # th = threading.Timer( 1, pr,)
        # th.daemon = True
        # th.start()






img = cv2.imread('ballon.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)  # auto resized
cv2.setMouseCallback('image2', mouse_callback)

while True:
    cv2.imshow('image2', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

#
# from util.calibration import Calibration
#
# calibration = Calibration()
#
# calibration_thread = threading.Thread(target=calibration.start_cali(), name='calibration_th2')
# calibration_thread.daemon = True
# calibration_thread.start()
#
# print("thread start")
# calibration_thread.join()
# print("wait thread")
#








# img = cv2.imread('ballon.jpg', cv2.IMREAD_GRAYSCALE)
#
# cv2.namedWindow('image2', cv2.WINDOW_NORMAL)  # auto resized
# cv2.setWindowProperty('image2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FREERATIO )
# cv2.imshow('image2', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# def a_thread():
#     cv2.namedWindow('image2', cv2.WINDOW_NORMAL)  # auto resized
#     cv2.setWindowProperty('image2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     print("thread start")
#     print("thread : imshow")
#     cv2.imshow('image2', np.zeros((300, 300, 3), np.uint8))
#     print("thread : waitkey")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("thread end")
#
#
# th = threading.Thread(target=a_thread, name='a_thr')
# th.daemon = True
# th.start()
#
#
# time.sleep(2)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # auto resized
# cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
# img = cv2.imread('ballon.jpg')
# img = cv2.imread('ballon.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()



## performance test했던코드


# from util.calibration import Calibration
#
# cali = Calibration()
# cali.Const_Display_X = 0
#
# ########### 추가 ##################
# import time  # time 라이브러리
#
# ###################################
# perf = Performance(1280, 720)
#
# cam = cv2.VideoCapture(-1)  # 카메라 생성
# # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# # 윈도우 생성 및 사이즈 변경
# var1 = 10
# var2 = 10
#
# cv2.namedWindow('CAM_Window')
# cv2.setMouseCallback('CAM_Window', perf.mouse_callback,param=(cali))
#
#
#
# prevTime = 0  # 이전 시간을 저장할 변수
# while (True):
#
#
#     cali.Const_Display_X = cali.Const_Display_X + 1
#
#     ret, perf.show_img = cam.read()
#
#     curTime = time.time()
#
#     sec = curTime - prevTime
#     prevTime = curTime
#     fps = 1 / (sec)
#
#
#     # 프레임 수를 문자열에 저장
#     str = "FPS : %0.1f" % fps
#
#     # 표시
#     cv2.putText(perf.show_img, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
#
#
#     if perf.save_index != 0 :
#         perf.draw_gaze_coordinate_mark()
#         perf.draw_real_coordinate_mark()
#     # 얻어온 이미지 윈도우에 표시
#     cv2.imshow('CAM_Window', perf.show_img)
#
#     # 10ms 동안 키입력 대기
#     if cv2.waitKey(10) >= 0:
#         break;
#
# # 윈도우 종려
# cam.release()
# cv2.destroyWindow('CAM_Window')
#


