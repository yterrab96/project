import struct

def send_one_message(s, data):
    length = len(data)
    s.sendall(struct.pack('!I', length))
    s.sendall(data)

def recv_one_message(s):
    lengthbuf = recvall(s, 4)
    if not lengthbuf:
        return None
    length, = struct.unpack('!I', lengthbuf)
    return recvall(s, length)

def recvall(s, count):
    buf = b''
    while count:
        newbuf = s.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf 
