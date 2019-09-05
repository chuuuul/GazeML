#-*- coding:utf-8 -*-

import socket
import threading
import time


def recieve(sock):
    while True:
        recvData = sock.recv(1024)
        print('상대방 :', recvData.decode('utf-8'))
        first,second = recvData.split(',')


# 접속 정보 설정
SERVER_IP = '127.0.0.1'
SERVER_PORT = 2349
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)

# 클라이언트 소켓 설정
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:



    client_socket.connect(SERVER_ADDR)  # 서버에 접속
    # client_socket.send('hi'.encode())  # 서버에 메시지 전송

    receiver = threading.Thread(target=recieve, args=(client_socket,))
    receiver.start()

    while True:
        time.sleep(1)
        pass