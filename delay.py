import control as ctrl
import numpy as np


def delay(L):
    num_delay, den_delay = ctrl.pade(L, 5)  # 5th-order Pade approximation
    delay_tf = ctrl.tf(num_delay, den_delay) 
    return delay_tf