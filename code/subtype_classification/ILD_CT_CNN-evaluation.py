#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import label_binarize


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"


# In[3]:


df_val = pd.read_excel("/raid/data/yanglab/ILD/joint_val_xm.xlsx")


# In[4]:


df_val.head()


# In[ ]:





# In[4]:


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)


# In[5]:


image_size = 256


# In[6]:


val_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        x_col = 'filename',
        y_col = 'Consensus',
        target_size=(image_size, image_size),
        batch_size=1,
        shuffle=False,
        seed=726,
        class_mode='raw')


# In[3]:


best_model = load_model("/raid/data/yanglab/ILD/ILD_results/models/ct-unfreezetop10-CT-IRV2-256-400-0.001.h5")
best_model.summary()
#pred = best_model.predict_generator(val_generator,steps=len(val_generator.labels), verbose=1)


# In[8]:


df_prob = pd.DataFrame({'MRN':df_val.MRN,'filename':val_generator.filenames, 'label':list(val_generator.labels)})
d_pred = pd.DataFrame(pred)
d_pred.columns = ["chp","nsip","o","sar",'uip']
df_prob = pd.concat([df_prob, d_pred], axis=1)


def get_series_name(f):
    x = f[:-8]
    return x


# In[9]:


df_prob.head()


# In[10]:


df_prob['series'] = df_prob.filename.apply(get_series_name)


# In[11]:


df_mean = df_prob.groupby(['MRN','label','series']).mean().reset_index()
df_mean


# In[12]:


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


# In[14]:


#df_mean.to_excel("/raid/data/yanglab/ILD/ILD_results/prob_sheet_validation/ct-cnn_val.xlsx", index=False)


# In[ ]:




