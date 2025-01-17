import socket


class GazeDataSender:

    def __init__(self, cali):
        # 통신 정보 설정
        self.IP = ''
        self.PORT = 23488
        self.SIZE = 1024
        self.CLIENT_ADDR = (self.IP, self.PORT)
        self.cali = cali

    def send_gaze(self):
        while True:
            try:

                print("소켓 준비 완료( port : ",self.PORT,")")
                # 서버 소켓 설정
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM,) as server_socket:
                    server_socket.bind(self.CLIENT_ADDR)    # 주소 바인딩
                    server_socket.listen()      # 클라이언트의 요청을 받을 준비

                    client_socket, client_addr = server_socket.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
                    print("소켓 연결 완료")

                    # 시선정보 전송 할 때
                    # 보낼 gaze 정보가 없으면 대기
                    # while self.cali.right_gaze_coordinate is None or self.cali.left_gaze_coordinate is None:
                    #     continue

                    # 시선영역 전성 할 때
                    while self.cali.current_point is None :
                        continue

                    while True:

                        #### 영역 전송 ####
                        send_msg = str(self.cali.current_point)
                        # print("보낼데이터 : ",send_msg)
                        client_socket.send( send_msg.encode() )
                        client_socket.recv(self.SIZE)     # 받았다는 신호 기다림

                        ### 맞은 패턴 전송 ###
                        send_msg = ""
                        if len(self.cali.correct_point) !=  0:
                            # 보낼때_배열->문자열
                            for i in self.cali.correct_point:
                                send_msg = str(send_msg) + str(i)
                        else:
                            send_msg = "0"          # 0 : 맞은패턴이없다.

                        client_socket.send( send_msg.encode() )
                        client_socket.recv(self.SIZE)     # 받았다는 신호 기다림


            except:
                print("소켓 닫힘")
                client_socket.close()  # 클라이언트 소켓 종료
