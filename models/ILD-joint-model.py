import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, activations
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
