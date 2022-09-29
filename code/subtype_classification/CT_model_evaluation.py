import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


df_test = pd.read_excel('./data/classification/test_joint.csv')
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)

image_size = 256

test_generator = data_generator.flow_from_dataframe(
        dataframe=df_test,
        x_col = 'filename',
        y_col = 'Consensus',
        target_size=(image_size, image_size),
        batch_size=1,
        shuffle=False,
        seed=726,
        class_mode='raw')

best_model = load_model("./models/classification-CT-CNN.h5") ### CT transformer model: "./models/classification-CT-transformer.h5"
pred = best_model.predict_generator(test_generator,steps=len(test_generator.labels), verbose=1)
