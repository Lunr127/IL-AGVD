import models
import time
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd

def build_model(model_name):
    if model_name == 'XCLIP':
        model = models.XCLIP()
    return model

