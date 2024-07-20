import os
import random
import glob
import gc  # garbage collector

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import cv2

# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.layers import (Conv2D, Input, MaxPooling2D, 
                                     Dropout, concatenate, UpSampling2D, BatchNormalization, Conv2DTranspose)

from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

# =====================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Please ensure CUDA and cuDNN are properly installed.")

# =====================
FULL_SHAPE = (768, 768)
NEW_SHAPE = (128, 128)

def image_from_path(img_dir, img_name):
    
    img_path = os.path.join(img_dir, img_name)
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, NEW_SHAPE)
    image = image / 255.0  # Normalize to [0, 1]
    
    return np.expand_dims(image, axis=0)






