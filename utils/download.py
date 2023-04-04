import os
import shutil
import sys
import time
from urllib import request
import tarfile
import zipfile


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


def download_to_dir(url, dst=None):
    file_name = os.path.split(url)[1]
    if dst is None:
        dst = file_name.split(".")[0]
    file_dir = os.path.join(dst, file_name)
    if os.path.exists(file_dir):
        sys.stdout.write(f"File exists")
        return file_dir
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)
    hook = reporthook()
    request.urlretrieve(url, file_dir, hook)
    sys.stdout.write(f"\n{file_name} saved at {file_dir}")
    return file_dir


def download(url, dst=""):
    file_name = os.path.split(url)[1]
    file_dir = os.path.join(dst, file_name)
    if os.path.exists(file_dir):
        sys.stdout.write(f"File exists")
        return file_dir
    hook = reporthook()
    request.urlretrieve(url, file_dir, hook)
    sys.stdout.write(f"\n{file_name} saved at {file_dir}\n")
    return file_dir
