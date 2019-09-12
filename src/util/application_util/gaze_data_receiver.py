#-*- coding:utf-8 -*-

import socket
import threading
import time

# Qt 클래스 필요함
class GazeDataReceiver:


    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 23488
    SIZE = 1024
    gaze_x = None
    gaze_y = None
    current_point = None
    correct_point = []

    def __init__(self,qt):
        self.qt = qt
        pass

    def receive_gaze(self):

        # 클라이언트 소켓 설정
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)
            client_socket.connect(SERVER_ADDR)  # 서버에 접속        # port 에러

            print("소켓 연결 성공")

            while True:

                #### 영역 수신 ####
                recvData = client_socket.recv(1024)
                recvData = recvData.decode('utf-8')

                self.current_point = int(recvData)
                # print("receive data : " , self.current_point)
                client_socket.send( "ok".encode() )
                # print(self.current_point)

                #### 맞은 패턴 수신 ####

                recvData = client_socket.recv(1024)
                recvData = recvData.decode('utf-8')

                # 받을때_문자열->_배열
                self.correct_point = []

                if recvData != "0" :
                    for i in range(len(recvData)):
                        self.correct_point.append(int(recvData[i]))

                # print("receive array : ", self.correct_point)
                client_socket.send( "ok2".encode() )


                # 화면_영역표시_업데이트
                self.qt.do_update()

                time.sleep(0.03)    # 너무 빨라서..






# gazeDataReceiver =  GazeDataReceiver()
# gazeDataReceiver.receive_gaze()