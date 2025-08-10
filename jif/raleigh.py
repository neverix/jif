# communicate with remote nodes

import threading
import queue
import socket
import time
import simple_parsing
from dataclasses import dataclass
from typing import Any


@dataclass
class RaleighInfo(simple_parsing.Serializable):
    ports: list[int]
    hosts: list[Any]
    seed: int
    params_seed: int = 0
    group_id: int = 0


class RaleighCommunicator(threading.Thread):
    def __init__(self, ports, friends=[]):
        super().__init__()
        self.ports = ports
        self.friends = list(map(tuple, friends))
        self.clients = {}
        self.servers = {}
        self.queue = queue.Queue()
        self.results_queue = queue.Queue()

    def run(self):
        for port in self.ports:
            self.servers[port] = RaleighFriendServer(port, self.friends[0], self.queue)
        for friend in self.friends:
            self.clients[friend] = RaleighFriendClient(friend, self.results_queue)
        
        for server in self.servers.values():
            server.start()
            server.started.wait()
        for client in self.clients.values():
            client.start()
        
        for server in self.servers.values():
            server.join()
        for client in self.clients.values():
            client.join()

    def __del__(self):
        for server in self.servers.values():
            server.server.close()
        for client in self.clients.values():
            client.client.close()


class RaleighFriendServer(threading.Thread):
    def __init__(self, port, friend, queue):
        super().__init__()
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.started = threading.Event()
        self.friend = friend
        self.queue = queue

    def run(self):
        self.server.bind(("0.0.0.0", self.port))
        self.server.listen(5)
        self.started.set()

        print(f"{self.port} waiting for connection from {self.friend}")
        while True:
            new_friend, addr = self.server.accept()
            print(f"Got connection from {addr}")
            if addr[0] != self.friend[0]:
                continue
            break
        print(f"{self.port} accepted connection from {addr}")
        assert new_friend.recv(1024) == b"hello"
        new_friend.send(b"hello hello")
        print(f"{self.port} sent handshake")

        while True:
            msg_type, msg = self.queue.get()
            match msg_type:
                case "last_qs":
                    new_friend.send(msg)
                case _:
                    raise ValueError(f"Unknown message type: {msg_type}")

class RaleighFriendClient(threading.Thread):
    def __init__(self, friend, results_queue, buffer_size=65536):
        super().__init__()
        self.friend = friend
        self.results_queue = results_queue
        self.client = None
        self.buffer_size = buffer_size

    def run(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                print(f"Trying to connect to {self.friend}")
                self.client.connect(self.friend)
                break
            except (TimeoutError, ConnectionRefusedError):
                time.sleep(1)
        print(f"Connected to {self.friend}")

        print(f"{self.friend} connecting")
        start_time = time.time()
        self.client.send(b"hello")
        assert self.client.recv(1024) == b"hello hello"
        print(f"{self.friend} connected in {time.time() - start_time} seconds")
        self.results_queue.put(("ready", self.friend))
        
        while True:
            sent_back = self.client.recv(self.buffer_size)
            self.results_queue.put(("last_qs", self.friend, sent_back))
