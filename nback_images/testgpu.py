import os
import path
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())