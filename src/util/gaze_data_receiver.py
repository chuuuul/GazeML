#-*- coding:utf-8 -*-

import socket
import threading
import time


class GazeDataReceiver:


    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 23487
    SIZE = 1024




    def __init__(self):
        pass

    def receive_gaze(self):

        # 클라이언트 소켓 설정
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            SERVER_ADDR = (self.SERVER_IP, self.SERVER_PORT)
            client_socket.connect(SERVER_ADDR)  # 서버에 접속
            # client_socket.send('hi'.encode())  # 서버에 메시지 전송

            # receiver = threading.Thread(target=receieve, args=(client_socket,))
            # receiver.start()

            # def receieve(sock):
            while True:
                recvData = client_socket.recv(1024)
                recvData = recvData.decode('utf-8')
                print('상대방 :', recvData)
                #꼬리에붙은 - 구분자 제거
                # data, trash = str(recvData).split('-', 1)
                first, second = str(recvData).split(',')
                print("first : ", first)
                print("second : ", second)
                client_socket.send( "ok".encode())

            while True:
                time.sleep(1)
                pass

gazeDataReceiver =  GazeDataReceiver()
gazeDataReceiver.receive_gaze()