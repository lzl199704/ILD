import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,Callback
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



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


df_train=pd.read_csv(args.train_path)
df_val=pd.read_csv(args.val_path)
df_test = pd.read_csv(args.test_path)


num_classes = 5
batch_size = 400
num_epoches = 40
learning_rate = 0.001
weight_decay = 0.0001
input_shape = (512, 512, 3)
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
        y_col = 'Consensus',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='raw')


validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        x_col = 'filename',
        y_col = 'Consensus',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='raw')


checkpoint_filepath="./models/classification-CT-CNN.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )



def get_compiled_model():
    model_dir ="../RadImageNet_models/RadImageNet-IRV2-notop.h5"
    base_model = InceptionResNetV2(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    y = base_model.output   
    y = Dropout(0.5)(y)
    y = Dense(1024, activation='softmax')(y)
    predictions = Dense(num_classes, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0001)
    model.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model


train_steps =  df_train.shape[0]/ batch_size
val_steps = df_val.shape[0] / batch_size


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



with strategy.scope():
    model = get_compiled_model()


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
