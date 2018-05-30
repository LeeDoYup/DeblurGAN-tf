from __future__ import print_function

import os
import sys
import logging
import js

import tensorflow as tf 
import numpy as np 
from models.cgan_model import cgan

def lineaer_decay(initial=0.0001, step, start_step=150, end_step=300):
    '''
    return decayed learning rate
    It becomes 0 at end_step
    '''
    decay_period = end_step - start_step
    step_decay = (initial-0.0)/decay_period
    update_step = max(0, step-start_step)
    current_value = max(0, initial - (update_step)*step_decay)
    return current_value


