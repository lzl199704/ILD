#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from sklearn.preprocessing import label_binarize


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"


# In[3]:


df_val = pd.read_excel("/raid/data/yanglab/ILD/joint_test_600.xlsx")


# In[4]:


train_variables = [
       'kvp','manufacturer','Home_O2','Occupation','CurrentSmoker','FormerSmoker','Hx_disease','sex','age',
          'pulm_HTN','Biopsy','FEV1','FVC','FEV1/FVC','DLCO','thickness']


# In[5]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()


# In[6]:


df_norm_val = pd.DataFrame(ss.fit_transform(df_val[train_variables]),columns = df_val[train_variables].columns)
df_norm_val['filename'] = df_val.filename
df_norm_val.head()


# In[7]:


num_classes = 5
batch_size = 1
num_epoches = 1
image_size = 256  # We'll resize input images to this size


# In[8]:


class ImageTextDataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""
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
            
    def __get_label(self, file_name):
        return int(self.labels[self.labels.index == file_name]['Consensus'])
        #return 0 if 'benigin' in file_name else 1

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
            #img = tf.convert_to_tensor(img, dtype=tf.float32)
            # Normalization
            for ch in range(self.n_channels):
                img[:, :, ch] = (img[:, :, ch] - self.ave[ch])/self.std[ch]
            
            if self.shuffle:
                # Some image augmentation codes
                ###### You can put your preprocessing codes here. #####
                #img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
                #img = tf.image.random_brightness(img, max_delta=0.5)
                #return img
                pass

            X_img.append(img)
            df_cf = self.clinical_info[self.clinical_info['filename']== img_file]
            df_cf.set_index('filename',inplace=True)
            c_info = df_cf.values.flatten()
            if len(c_info) < 3:
                print(filename)
            X_clinical.append(c_info)
            y[i] = self.__get_label(img_file)
            
        X = [np.array(X_img), np.array(X_clinical)]
        #print(y)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# In[9]:


val_labels = df_val[['Consensus']]
val_labels['filename'] = df_norm_val['filename']
val_labels.set_index('filename', inplace=True)


# In[10]:


val_labels


# In[11]:


val_datagen = ImageTextDataGenerator(img_files=list(df_norm_val.filename), 
                                     clinical_info=df_norm_val, 
                                     dim = (image_size, image_size), 
                                     labels=val_labels,
                                     batch_size=1,
                                      shuffle=False
                                     )


# In[12]:


from tensorflow.keras.models import load_model


# In[13]:


best_model = model = load_model("/raid/data/yanglab/ILD/ILD_results/models/joint_no_med-600-unfreezetop10-CT-IRV2-256-400-0.0001.h5", compile=False)

pred = best_model.predict(val_datagen, steps=df_norm_val.shape[0], verbose=1)


# In[14]:


best_model.summary()


# In[15]:


pred


# In[16]:


df_prob = pd.DataFrame({'MRN':df_val.MRN,'filename':df_val.filename, 'label':list(df_val.Consensus)})
d_pred = pd.DataFrame(pred)
d_pred.columns = ["chp","nsip","o","sar",'uip']
df_prob = pd.concat([df_prob, d_pred], axis=1)


def get_series_name(f):
    x = f[:-8]
    return x


# In[17]:


df_prob['series'] = df_prob.filename.apply(get_series_name)


# In[18]:


df_mean = df_prob.groupby(['MRN','label','series']).mean().reset_index()


# In[19]:


df_mean


# In[20]:


n_classes=5
fpr = dict()
tpr = dict()
roc_auc = dict()
pred_per_patient = np.array(df_mean.iloc[:, 3:])
y_true=label_binarize(df_mean.label, classes=[0, 1, 2, 3,4])
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:,i], pred_per_patient[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(i, roc_auc[i])


# In[24]:


df_mean.to_excel("/raid/data/yanglab/ILD/ILD_results/prob_sheet/joint_cnn_test_nomed_unfreezetop10_v0819.xlsx", index=False)


# In[22]:


y_pred = np.argmax(pred_per_patient, axis=1)


# In[23]:


accuracy_score(df_mean.label, y_pred)

