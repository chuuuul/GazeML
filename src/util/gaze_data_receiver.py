#-*- coding:utf-8 -*-

import socket
import threading
import time


class GazeDataReceiver:


    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 23487
    SIZE = 1024
    gaze_x = None
    gaze_y = None




    def __init__(self,qt):
        self.qt = qt
        pass

    def receive_gaze(self):

        # 클라이언트 소켓 설정
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)
            client_socket.connect(SERVER_ADDR)  # 서버에 접속        # port 에러
            # client_socket.send('hi'.encode())  # 서버에 메시지 전송

            # receiver = threading.Thread(target=receieve, args=(client_socket,))
            # receiver.start()

            # def receieve(sock):
            while True:
                recvData = client_socket.recv(1024)
                recvData = recvData.decode('utf-8')
                # print('상대방 :', recvData)
                #꼬리에붙은 - 구분자 제거
                # data, trash = str(recvData).split('-', 1)
                first , second = str(recvData).split(',')

                self.gaze_x = float (first)
                self.gaze_y = float (second)
                self.qt.do_update()
                client_socket.send( "ok".encode())
                time.sleep(0.05)    # 너무 빨라서..



# gazeDataReceiver =  GazeDataReceiver()
# gazeDataReceiver.receive_gaze()