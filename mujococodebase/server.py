import logging
import socket
from select import select
from mujococodebase.world_parser import WorldParser

logger = logging.getLogger(__file__)


class Server:
    def __init__(self, host: str, port: int, world_parser: WorldParser):
        self.world_parser: WorldParser = world_parser
        self.__host: str = host
        self.__port: str = port
        self.__socket: socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.__send_buff = []
        self.__rcv_buffer_size = 1024
        self.__rcv_buffer = bytearray(self.__rcv_buffer_size)

    def connect(self) -> None:
        logger.info("Connecting to server at %s:%d...", self.__host, self.__port)
        while True:
            try:
                self.__socket.connect((self.__host, self.__port))
                break
            except ConnectionRefusedError:
                logger.error(
                    "Connection refused. Make sure the server is running and listening on {self.__host}:{self.__port}."
                )

        logger.info(f"Server connection established to {self.__host}:{self.__port}.")

    def shutdown(self) -> None:
        self.__socket.close()
        self.__socket.shutdown(socket.SHUT_RDWR)

    def send_immediate(self, msg: str) -> None:
        """
        Sends only the desired message, without the buffer
        """
        try:
            self.__socket.send(
                (len(msg)).to_bytes(4, byteorder="big") + msg.encode()
            )  # Add message length in the first 4 bytes
        except BrokenPipeError:
            print("\nError: socket was closed by rcssserver3d!")
            exit()

    def send(self) -> None:
        """
        Send all committed messages
        """
        if len(select([self.__socket], [], [], 0.0)[0]) == 0:
            self.send_immediate(("".join(self.__send_buff)))
        else:
            logger.info("Server_Comm.py: Received a new packet while thinking!")
        self.__send_buff = []

    def commit(self, msg: str) -> None:
        """
        Appends message to buffer
        """
        self.__send_buff.append(msg)

    def commit_and_send(self, msg: str = "") -> None:
        """
        Appends a message to buffer and then sends the buffer
        """
        self.commit(msg)
        self.send()

    def receive(self) -> None:
        """
        Receive the next message from the TCP/IP socket and updates world
        """

        # Receive message length information
        if (
            self.__socket.recv_into(
                self.__rcv_buffer, nbytes=4, flags=socket.MSG_WAITALL
            )
            != 4
        ):
            raise ConnectionResetError

        msg_size = int.from_bytes(self.__rcv_buffer[:4], byteorder="big", signed=False)

        # Ensure receive buffer is large enough to hold the message
        if msg_size > self.__rcv_buffer_size:
            self.__rcv_buffer_size = msg_size
            self.__rcv_buffer = bytearray(self.__rcv_buffer_size)

        # Receive message with the specified length
        if (
            self.__socket.recv_into(
                self.__rcv_buffer, nbytes=msg_size, flags=socket.MSG_WAITALL
            )
            != msg_size
        ):
            raise ConnectionResetError

        self.world_parser.parse(message=self.__rcv_buffer[:msg_size].decode())

    def commit_beam(self, pos2d: list, rotation: float) -> None:
        assert len(pos2d) == 2
        self.commit(f"(beam {pos2d[0]} {pos2d[1]} {rotation})")
