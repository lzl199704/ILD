import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Input

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./data/classification/train_joint.csv')
parser.add_argument('--val_path', type=str, default='./data/classification/val_joint.csv')
parser.add_argument('--test_path', type=str, default='./data/classification/test_joint.csv')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

###Build Vit model as following: https://keras.io/examples/vision/image_classification_with_vision_transformer/
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size,**kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({
            'patch_size': self.patch_size       
        })
        return config
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config['num_pathes'] = self.num_patches
        config['projection_dim'] = self.projection_dim
        return config
  
def ct_model():
    ## create vit
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    x = Dense(1024, activation='relu')(features)
    x = Dense(32, activation='relu')(x)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def get_compiled_model():
    model = ct_model()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0001)
    model.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model  
  
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

patch_size = 10  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] 

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


checkpoint_filepath="./models/classification-CT-transformer.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )

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
d_loss.to_excel("./loss/loss_classification_CT_transformer.xlsx", index=False)
