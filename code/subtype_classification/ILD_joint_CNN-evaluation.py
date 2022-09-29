import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, activations
from tensorflow.keras.models import load_model
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, top_k_accuracy_score
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

###parameters
num_classes = 5
batch_size = 1
num_epoches = 1
image_size = 256  # We'll resize input images to this size
       
###evaluation on test dataset
df_test = pd.read_excel('./data/classification/test_joint.csv')
train_variables = [
       'kvp','manufacturer','Home_O2','Occupation','CurrentSmoker','FormerSmoker','Hx_disease','sex','age',
          'pulm_HTN','Biopsy','FEV1','FVC','FEV1/FVC','DLCO','thickness']

ss = StandardScaler()
df_norm_test = pd.DataFrame(ss.fit_transform(df_test[train_variables]),columns = df_test[train_variables].columns)
df_norm_test['filename'] = df_test.filename

test_labels = df_test[['Consensus']]
test_labels['filename'] = df_norm_test['filename']
test_labels.set_index('filename', inplace=True)


test_datagen = ImageTextDataGenerator(img_files=list(df_norm_test.filename), 
                                     clinical_info=df_norm_test, 
                                     dim = (image_size, image_size), 
                                     labels=test_labels,
                                     batch_size=1,
                                      shuffle=False
                                     )


best_model = model = load_model("./models/classification-joint-CNN.h5", compile=False)
pred = best_model.predict(test_datagen, steps=df_norm_test.shape[0], verbose=1)
