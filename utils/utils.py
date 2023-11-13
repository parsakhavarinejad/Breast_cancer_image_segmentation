import os
import glob
import pandas as pd
import numpy as np

def load_data(path):
    masks = glob.glob("data/Dataset_BUSI_with_GT/*/*_mask.png")
    images = [mask_images.replace("_mask", "") for mask_images in masks]

