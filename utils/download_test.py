import socket
import sys
import time
from urllib import request


def reporthook():
    start_time = 0

    def show_progress(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
        duration = time.time() - start_time
        duration = 1 if duration == 0 else duration
        progress_size = int(count * block_size)
        speed = progress_size / (1024.0 ** 2 * duration)
        percent = count * block_size * 100.0 / total_size
        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024. ** 2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()

    return show_progress

hook = reporthook()

ip = socket.gethostbyname(socket.gethostname())
proxy = request.ProxyHandler({'http': ip, 'https': ip})
source_address = "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip"
request.urlretrieve(source_address, "./data.tar", hook)
