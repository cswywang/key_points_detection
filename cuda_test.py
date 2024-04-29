from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())   # 抓取到GPU相关信息则可以使用GPU
