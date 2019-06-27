from enum import Enum, auto
import math


class ActivationFunctions(Enum):
    TanH = auto()


def activate(function: ActivationFunctions, value):
    if(function == ActivationFunctions.TanH):
        return math.tanh(value)
