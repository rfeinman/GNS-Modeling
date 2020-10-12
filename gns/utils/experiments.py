"""
utilities for experiments
"""
import os
import datetime


def time_string(seconds):
    seconds = int(round(seconds))
    return str(datetime.timedelta(seconds=seconds))

def mkdir(dirpath):
    if os.path.exists(dirpath):
        return
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        pass