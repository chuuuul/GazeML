import socket


class GazeDataSender:

    def __init__(self, cali):
        # 통신 정보 설정
        self.IP = ''
        self.PORT = 23487
        self.SIZE = 1024
        self.CLIENT_ADDR = (self.IP, self.PORT)
        self.cali = cali

    def send_gaze(self):
        print("send_gaze thread start")
        # 서버 소켓 설정
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM,) as server_socket:
            server_socket.bind(self.CLIENT_ADDR)    # 주소 바인딩
            server_socket.listen()      # 클라이언트의 요청을 받을 준비


            client_socket, client_addr = server_socket.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
            print("socket connect")
            # 보낼 gaze 정보가 없으면 대기
            while self.cali.right_gaze_coordinate is None or self.cali.left_gaze_coordinate is None:
                continue

            print("there is gaze info")
            while True:
                gaze_coordination = (self.cali.left_gaze_coordinate + self.cali.right_gaze_coordinate) / 2.0
                print()
                # 꼬리에 - 붙여서 구분자
                send_msg = str(gaze_coordination[0])+","+str(gaze_coordination[1])


                print("보낼데이터 : ",send_msg)
                client_socket.send( send_msg.encode() )  # 클라이언트에게 응답

                msg = client_socket.recv(self.SIZE) # 버퍼가 쌓이지 않도록 상대편에서 신호를 줄 때 까지 대기

            client_socket.close()  # 클라이언트 소켓 종료
