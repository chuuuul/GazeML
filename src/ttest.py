# Tensorflow 확 인코드
# import tensorflow as tf
# hello = tf.constant('hello tensol')
# sess = tf.Session()
# print(sess.run(hello))


print (3.144444444/3)
# import threading
# import numpy as np
# import cv2
# import time
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


