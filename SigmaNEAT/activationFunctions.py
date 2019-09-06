import math
from config import cudaMethod
ACTIVATION_FUNCTION__TANH = 0


@cudaMethod()
def activate(function: int, value):
    if(function == ACTIVATION_FUNCTION__TANH):
        return math.tanh(float(value))
