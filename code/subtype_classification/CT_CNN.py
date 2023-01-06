import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import roc_auc_score

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./data/classification/train_joint.csv')
parser.add_argument('--val_path', type=str, default='./data/classification/val_joint.csv')
parser.add_argument('--test_path', type=str, default='./data/classification/test_joint.csv')
args = parser.parse_args()

df_train=pd.read_csv(args.train_path)
df_val=pd.read_csv(args.val_path)
df_test = pd.read_csv(args.test_path)

num_classes = 5
batch_size = 400
num_epoches = 40
learning_rate = 0.001
image_size = 256 

train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)

train_generator = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        x_col = 'filename',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='raw')

validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        x_col = 'filename',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='raw')

checkpoint_filepath="./models/classification-CT-CNN.h5"
checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )

###RadImageNet pretrained models can be downloaded at https://github.com/BMEII-AI/RadImageNet
def get_compiled_model():
    model_dir ="../RadImageNet_models/CT-IRV2-notop.h5"
    base_model = InceptionResNetV2(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    for layer in base_model.layers[:-10]:
        layer.trainable = False 
    y = base_model.output
    predictions = Dense(num_classes, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0001)
    model.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

model = get_compiled_model()

train_steps =  df_train.shape[0]/ batch_size
val_steps = df_val.shape[0] / batch_size


history = model.fit(
        train_generator,
        epochs=num_epoches,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=val_steps,
        use_multiprocessing=True,
        callbacks=[checkpoint_callback])

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
d_loss = pd.DataFrame({'train_acc':train_acc, 'val_acc':val_acc, 'train_loss':train_loss, 'val_loss':val_loss})
d_loss.to_excel("./loss/loss_classification_CT_CNN.xlsx", index=False)
