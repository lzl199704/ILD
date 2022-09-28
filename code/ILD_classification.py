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

import argparse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_node', type=str, help='specify gpu nodes')
parser.add_argument('--train_path', type=str, default='./data/classification/train_joint.csv')
parser.add_argument('--val_path', type=str, default='./data/classification/val_joint.csv')
parser.add_argument('--batch_size', type=int, help='batch size', default=256)
parser.add_argument('--image_size', type=int, help='image size', default=256)
parser.add_argument('--epoch', type=int, help='number of epochs', default=30)
parser.add_argument('--structure', type=str, help='unfreezeall/freezeall/unfreezetop10', default=30)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_node

num_classes = 5

if not args.structure in ['unfreezeall', 'freezeall','unfreezetop10']:
    raise Exception('Freeze any layers? Choose to unfreezeall/freezeall/unfreezetop10 layers for the network.')


image_size = args.image_size
batch_size = args.batch_size
num_epoches = args.epoch

df_t=pd.read_csv(args.train_path)
df_v=pd.read_csv(args.val_path)

train_variables = [
       'kvp','manufacturer','Home_O2','Occupation','CurrentSmoker','FormerSmoker','Hx_disease','sex','age',
          'pulm_HTN','Biopsy','FEV1','FVC','FEV1/FVC','DLCO','thickness']

ss = StandardScaler()
df_norm_train = pd.DataFrame(ss.fit_transform(df_t[train_variables]),columns = df_t[train_variables].columns)
df_norm_train['filename'] = df_t.filename
df_norm_val = pd.DataFrame(ss.fit_transform(df_v[train_variables]),columns = df_v[train_variables].columns)
df_norm_val['filename'] = df_v.filename


def get_compiled_model():
    meta_input = Input(shape=(df_norm_train.shape[1]-1,), name='text') # 20 is the number of clinical variables
    meta = Dense(64,activation='relu')(meta_input) ## this is the MLP to train clinical variables
    meta = Dense(32,activation='relu')(meta)
    
    model_dir ="../RadImageNet_models/RadImageNet-IRV2-notop.h5" ###RadImageNet pretrained models can be downloaded at https://github.com/BMEII-AI/RadImageNet
    base_model = InceptionResNetV2(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    if args.structure == 'freezeall':
        for layer in base_model.layers:
            layer.trainable = False
    if args.structure == 'unfreezeall':
        pass
    if args.structure == 'unfreezetop10':
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    
    concatenated = Concatenate(axis=-1)([x, meta]) ## combine image and cv
    concat = Dropout(0.5)(concatenated)

    logits = Dense(num_classes, activation='softmax')(concat)
    my_new_model = Model(inputs=[base_model.input,meta_input], outputs=logits)

    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, decay=0.0001)
    my_new_model.compile(optimizer=adam, loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return my_new_model


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()

train_steps =  df_norm_train.shape[0]/ batch_size
val_steps = df_norm_val.shape[0] / batch_size

class ImageTextDataGenerator(keras.utils.Sequence):
    """Generates data for joint input."""
    def __init__(self, 
                 img_files=None, 
                 clinical_info=None, 
                 labels=None, 
                 ave=None, 
                 std=None, 
                 batch_size=32, 
                 dim=(300, 300), 
                 n_channels=3,
                 n_classes=num_classes, 
                 shuffle=True):
        """Initialization.
        Args:
            img_files: A list of path to image files.
            clinical_info: A dictionary of corresponding clinical variables.
            labels: A dictionary of corresponding labels.
        """
        self.img_files = img_files
        self.clinical_info = clinical_info
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        if ave is None:
            self.ave = np.zeros(n_channels)
        else:
            self.ave = ave
        if std is None:
            self.std = np.zeros(n_channels) + 1
        else:
            self.std = std
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(img_files_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_files_temp):
        """Generates data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # X = [np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))]
        X_img = []
        X_clinical = []
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, img_file in enumerate(img_files_temp):
            # Read image
            img = image.load_img(img_file, target_size=self.dim)
            
            # Convert to 3 channels
            img = image.img_to_array(img)
            
            # Preprocess_input and rescale images 
            img = preprocess_input(img)
            img = img/255.0 
            
            # Normalization
            for ch in range(self.n_channels):
                img[:, :, ch] = (img[:, :, ch] - self.ave[ch])/self.std[ch]
           
            X_img.append(img)
            df_cf = self.clinical_info[self.clinical_info['filename']== img_file]
            df_cf.set_index('filename',inplace=True)
            c_info = df_cf.values.flatten()
            if len(c_info) < 3:
                print(filename)
            X_clinical.append(c_info)
            y[i] = int(self.labels[self.labels.filename == img_file]['Consensus'])
        X = [np.array(X_img), np.array(X_clinical)]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

filepath="./models/classification-joint-CNN.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

train_labels = df_t[['Consensus']]
val_labels = df_v[['Consensus']]

train_labels['filename'] = df_norm_train['filename']
val_labels['filename'] = df_norm_val['filename']

train_datagen = ImageTextDataGenerator(img_files=list(df_norm_train.filename), 
                              clinical_info=df_norm_train,
                              dim=(image_size, image_size),
                              labels=train_labels,  
                              batch_size=batch_size, 
                                       shuffle=True)
val_datagen = ImageTextDataGenerator(img_files=list(df_norm_val.filename), 
                                     clinical_info=df_norm_val, 
                                     dim = (image_size, image_size), 
                                     labels=val_labels,
                                     batch_size=batch_size, 
                                     shuffle=True)

history = model.fit(
        train_datagen,
        epochs=num_epoches,
        steps_per_epoch=train_steps,
        validation_data=val_datagen,
        validation_steps=val_steps,
        use_multiprocessing=True,
        callbacks=[checkpoint])

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
d_loss = pd.DataFrame({'train_acc':train_acc, 'val_acc':val_acc, 'train_loss':train_loss, 'val_loss':val_loss})
d_loss.to_excel("./loss/loss_classification_joint_CNN.xlsx", index=False)

