import atexit
import subprocess
import os
import glob
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

IP = "127.0.0.1"
PORT_FROM_MAX = 54673


def send_audio():
    print('sharing audio')
    subprocess.run(["scp", "voice.wav", "richard@10.234.21.13:~/rapping-robot"])
    print('sharing txt')
    subprocess.run(["scp", "tiny.txt", "richard@10.234.21.13:~/rapping-robot"])
    return


def clear_chunks():
    files = glob.glob('/Users/richardsavery/Desktop/chunks/*')
    print('delete chunks')
    for f in files:
        os.remove(f)

def interactive(name, *args):
    print('message recieved from max')
    max_message = list(args)
    send_audio()


# OSC Listening
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/max", interactive)

server = osc_server.ThreadingOSCUDPServer((IP, PORT_FROM_MAX), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()

# close server when finished
atexit.register(server.server_close())