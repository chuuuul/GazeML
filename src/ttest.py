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


