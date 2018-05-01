# https://github.com/jonsafari/nvidia-ml-py/blob/master/pynvml.py
import argparse
import platform

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client import start_http_server, core

from time import sleep
from pynvml import *


# Getting the hostname
hostname = platform.node()

# Argparser
parser = argparse.ArgumentParser(description='nVidia GPU Prometheus Metrics Exporter')
parser.add_argument('--host', default="localhost")
parser.add_argument('--port', default="9091")
args = parser.parse_args()

gateway = ("{0:s}:{1:s}".format(args.host, args.port))

#setting labels 
registry = CollectorRegistry()
GPU_Total_Memory = Gauge('gpu_total_memory', 'Total installed frame buffer memory', ['device'], registry=registry)
GPU_Free_Memory = Gauge('gpu_free_memory','Unallocated frame buffer memory', ['device'], registry=registry)
GPU_Used_Memory = Gauge('gpu_used_memory', 'Allocated frame buffer memory ', ['device'], registry=registry)
gpu_utilization = Gauge('gpu_utilization_pct', 'Percent of time over the past sample period during which one or more kernels was executing on the GPU.', ['device'], registry=registry)
memory_utilization = Gauge('gpu_mem_utilization_pct', 'Percent of time over the past sample period during which global (device) memory was being read or written', ['device'], registry=registry)


def MemoryInfo(devID):
    MemoryInfo = nvmlDeviceGetMemoryInfo(devID)
    GPU_Total_Memory.labels(device=devID).set(MemoryInfo.total/1024)
    GPU_Free_Memory.labels(device=devID).set(MemoryInfo.free/1024)
    GPU_Used_Memory.labels(device=devID).set(MemoryInfo.used/1024)
    return (MemoryInfo)


def MemoryUtalization(dev,devID):
    try:
        gpu_utilization.labels(device=dev).set(nvmlDeviceGetUtilizationRates(devID).gpu / 100.0)
        memory_utilization.labels(device=dev).set(nvmlDeviceGetUtilizationRates(devID).memory / 100.0)
        return ('Utilization statistics = {:s}', str(nvmlDeviceGetUtilizationRates(devID)))
    except NVMLError_NotSupported as e:
        return ("Utilization Not Supported")


def PushTo_Gateway(sleepy):
    if "localhost" not in  gateway:
        print ('Pushing metrics to gateway at {:s}...'.format(gateway))
        push_to_gateway(gateway, job=hostname, registry=registry)
        print ('Push complete.\n')
        sleep(sleepy)

    else:
        print ('Pushing metrics to gateway at {:s}...'.format("localhost:9091"))
        push_to_gateway("localhost:9091", job=hostname, registry=registry)
        print ('Push complete.\n')
        sleep(sleepy)



nvmlInit()
print('Nvidia Driver version: {:s}'.format(nvmlSystemGetDriverVersion()))
print('[+] {:d} devices found.'.format(nvmlDeviceGetCount()))

if __name__ == "__main__":
    try:
       while True:
            for Device in range(nvmlDeviceGetCount()):
                DeviceID = nvmlDeviceGetHandleByIndex(Device)
                print('Device handle for {:d} is {:s}'.format(Device, str(DeviceID)))

                print MemoryInfo(DeviceID)
                print MemoryUtalization(Device ,DeviceID)
                PushTo_Gateway(10)
    except KeyboardInterrupt as err:
        nvmlShutdown()
        print ("Got Ctrl+C, Shutting Down NVML")
        print (err)

