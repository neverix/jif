# communicate with remote nodes
PORT = 13370

import threading
import queue
import socket


class RaleighCommunicator(threading.Thread):
    def __init__(self, friends=[]):
        super().__init__()
        self.server = None
        self.friends = friends
        self.clients = {}
        self.servers = {}
        self.queue = queue.Queue()

    def run(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("0.0.0.0", PORT))
        self.server.listen(5)
        
        for friend in self.friends:
            self.clients[friend] = RaleighFriend(friend)
            self.clients[friend].start()
        
        for friend in self.friends:
            new_friend, addr = self.server.accept()
            addr = addr[0]
            assert addr in self.friends
            assert addr not in self.servers
            self.servers[addr] = new_friend

        while True:
            msg_type, msg = self.queue.get()
            match msg_type:
                case _:
                    pass
        
        # while True:
        #     msg_type, msg = self.queue.get()
        #     match msg_type:
        #         case ""
        

class RaleighFriend(threading.Thread):
    def __init__(self, friend):
        super().__init__()
        self.friend = friend
        self.client = None
        self.server = None

    def run(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.friend, PORT))

        self.client.send(b"hello")
