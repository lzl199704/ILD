import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers, activations
import os
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from time import time
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_label(f):
    l = []
    for i in f:
        if i=='D':
            l.append(1)
        else:
            l.append(0)
    #np.zero a numpy array
    return l

year_num=3

if year_num==0:
    shape_num=1
elif year_num==1:
    shape_num=4
elif year_num==2:
    shape_num=6
else: shape_num=7


df_train=pd.read_csv("/raid/nfs4/home/liuz14/projects/ILD/verification/time_series/train_"+str(year_num)+"year_original_0214.csv")
df_val=pd.read_csv("/raid/nfs4/home/liuz14/projects/ILD/verification/time_series/val_"+str(year_num)+"year_original_0214.csv")
df_test=pd.read_csv("/raid/nfs4/home/liuz14/projects/ILD/verification/time_series/test_"+str(year_num)+"year_original_0214.csv")



dff_train=df_train.drop(columns=['MRN','LivingStatus','Med_label','Therapeutic_label'])
dff_val=df_val.drop(columns=['MRN','LivingStatus','Med_label','Therapeutic_label'])
dff_test=df_test.drop(columns=['MRN','LivingStatus','Med_label','Therapeutic_label'])


def preprocess_dataframe(dataframe1,scaled_data):
    pid_list=scaled_data.MRN.unique()
    feature_list=scaled_data.columns
    for i in pid_list:
        subframe=scaled_data[scaled_data.MRN==i]
        if len(subframe.MRN)<(shape_num+1):
            a=subframe.LivingStatus.iloc[0]
            b=subframe.MRN.iloc[0]
            d = pd.DataFrame(0, index=np.arange(shape_num-len(subframe.MRN)), columns=feature_list)
            subframe=subframe.append(d)
            subframe.LivingStatus=a
            subframe.MRN=b
        else:
            subframe=subframe.iloc[0:shape_num]
        dataframe1=dataframe1.append(subframe)
    dataframe1=dataframe1.reset_index(drop=False)
    dataframe1.drop(['index'], axis=1, inplace=True)
    return dataframe1



ss=MinMaxScaler()
train_scaled = pd.DataFrame(ss.fit_transform(dff_train),columns = dff_train.columns)
val_scaled = pd.DataFrame(ss.fit_transform(dff_val),columns = dff_val.columns)
train_scaled['MRN']=df_train['MRN']
train_scaled['LivingStatus']=df_train['LivingStatus']
val_scaled['MRN']=df_val['MRN']
val_scaled['LivingStatus']=df_val['LivingStatus']

df1=pd.DataFrame()
df1=preprocess_dataframe(df1,train_scaled)
df2=pd.DataFrame()
df2=preprocess_dataframe(df2,val_scaled)

df_train_feature=df1.iloc[:,:(df1.shape[1]-2)]
df_val_feature=df2.iloc[:,:(df2.shape[1]-2)]
df_train_label=df1[['MRN','LivingStatus']]
df_val_label=df2[['MRN','LivingStatus']]

features1=df_train_feature.to_numpy()
x_train=np.reshape(features1,(len(df_train.MRN.unique()),shape_num,df_train_feature.shape[1]))
features2=df_val_feature.to_numpy()
x_val=np.reshape(features2,(len(df_val.MRN.unique()),shape_num,df_val_feature.shape[1]))



df_train_label=df_train_label.drop_duplicates(subset=['MRN','LivingStatus'], keep='first')
df_train_label=df_train_label.reset_index(drop=True)
df_val_label=df_val_label.drop_duplicates(subset=['MRN','LivingStatus'], keep='first')
df_val_label=df_val_label.reset_index(drop=True)

y_train = np.array(get_label(df_train_label["LivingStatus"]))
y_val = np.array(get_label(df_val_label["LivingStatus"]))


test_scaled = pd.DataFrame(ss.fit_transform(dff_test),columns = dff_test.columns)

test_scaled['MRN']=df_test['MRN']
test_scaled['LivingStatus']=df_test['LivingStatus']

df3=pd.DataFrame()
df3=preprocess_dataframe(df3,test_scaled)

df_test_feature=df3.iloc[:,:(df3.shape[1]-2)]

df_test_label=df3[['MRN','LivingStatus']]

features3=df_test_feature.to_numpy()
x_test=np.reshape(features3,(len(df_test.MRN.unique()),shape_num,df_test_feature.shape[1]))


df_test_label=df_test_label.drop_duplicates(subset=['MRN','LivingStatus'], keep='first')
df_test_label=df_test_label.reset_index(drop=True)
y_test = np.array(get_label(df_test_label["LivingStatus"]))



n_classes = len(np.unique(y_train))


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res



def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    metrics=["sparse_categorical_accuracy"],
                    )
    return model



input_shape = x_train.shape[1:]
num_heads = 4
filepath = "/raid/data/yanglab/ILD/ILD_results/models/transformer_without_treatment/nh4_year3_withoutmed_" 
    


def run_model(i):
    model = build_model(
        input_shape,
        head_size=512,
        num_heads=num_heads,
        ff_dim=16,
        num_transformer_blocks=16,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )
    
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath+str(i), monitor='val_loss', mode='min', save_weights_only=True, save_best_only=True, verbose=1
    )
    
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val,y_val),
        epochs=100,
        batch_size=64,
        callbacks=[checkpoint]
    )    
    
###run 30 iterations
for i in range(30):
    run_model(i)



