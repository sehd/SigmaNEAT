import numba.cuda as cu


@cu.jit
def kernel():
    print("hi")


kernel[100, 8]()
