import ctypes

def test_opencl():
    try:
        ocl = ctypes.CDLL('libOpenCL.so.1')
        num_platforms = ctypes.c_uint32()
        ocl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
        print(f"OpenCL platforms found: {num_platforms.value}")
        platforms = (ctypes.c_void_p * num_platforms.value)()
        ocl.clGetPlatformIDs(num_platforms.value, platforms, None)
        for i, p in enumerate(platforms):
            name = (ctypes.c_char * 256)()
            ocl.clGetPlatformInfo(p, 0x0902, 256, name, None)
            print(f"Platform {i}: {name.value.decode()}")
            num_devices = ctypes.c_uint32()
            res = ocl.clGetDeviceIDs(p, 0xFFFFFFFF, 0, None, ctypes.byref(num_devices))
            print(f"  Devices count: {num_devices.value} (res={res})")
            if num_devices.value > 0:
                devices = (ctypes.c_void_p * num_devices.value)()
                ocl.clGetDeviceIDs(p, 0xFFFFFFFF, num_devices.value, devices, None)
                for d in devices:
                    dname = (ctypes.c_char * 256)()
                    dtype = ctypes.c_uint64()
                    ocl.clGetDeviceInfo(d, 0x102B, 256, dname, None)
                    ocl.clGetDeviceInfo(d, 0x1000, 8, ctypes.byref(dtype), None)
                    t_str = "GPU" if (dtype.value & 0x4) else ("CPU" if (dtype.value & 0x2) else "Other")
                    print(f"    Device: {dname.value.decode()} [{t_str}]")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_opencl()
