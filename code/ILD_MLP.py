import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./data/classification/train_joint.csv')
parser.add_argument('--val_path', type=str, default='./data/classification/val_joint.csv')
parser.add_argument('--test_path', type=str, default='./data/classification/test_joint.csv')
args = parser.parse_args()

df_t=pd.read_csv(args.train_path)
df_v=pd.read_csv(args.val_path)
df_test = pd.read_csv(args.test_path)


df_test.MRN=df_test.MRN.astype(str)
df_t.MRN=df_t.MRN.astype(str)
df_test=df_test.reset_index(drop=True)


ss = StandardScaler()


df2_train=df_t.drop_duplicates(subset='MRN',keep='first')
df2_val=df_v.drop_duplicates(subset='MRN',keep='first')
df2_test=df_test.drop_duplicates(subset='MRN',keep='first')
df2_train=df2_train.reset_index(drop=True)
df2_val=df2_val.reset_index(drop=True)
df2_test=df2_test.reset_index(drop=True)



x_train=df2_train[[
       'kvp','manufacturer','Home_O2','Occupation','CurrentSmoker','FormerSmoker','Hx_disease','sex','age',
          'pulm_HTN','Biopsy','FEV1','FVC','FEV1/FVC','DLCO','thickness']]
y_train=df2_train['Consensus']

x_val=df2_val[[
       'kvp','manufacturer','Home_O2','Occupation','CurrentSmoker','FormerSmoker','Hx_disease','sex','age',
          'pulm_HTN','Biopsy','FEV1','FVC','FEV1/FVC','DLCO','thickness']]
y_val=df2_val['Consensus']

x_test=df2_test[[
       'kvp','manufacturer','Home_O2','Occupation','CurrentSmoker','FormerSmoker','Hx_disease','sex','age',
          'pulm_HTN','Biopsy','FEV1','FVC','FEV1/FVC','DLCO','thickness']]
y_test=df2_test['Consensus']



x_test = pd.DataFrame(ss.fit_transform(x_test),columns = x_test.columns)
x_train = pd.DataFrame(ss.fit_transform(x_train),columns = x_train.columns)
x_val = pd.DataFrame(ss.fit_transform(x_val),columns = x_val.columns)

n_classes=5
y1_mlp=y_train
y1_mlp=label_binarize(y1_mlp, classes=[0, 1, 2, 3,4])

y2_mlp=y_val
y2_mlp=label_binarize(y2_mlp, classes=[0, 1, 2, 3,4])

y3_mlp=y_test
y3_mlp=label_binarize(y3_mlp, classes=[0, 1, 2, 3,4])



random_state = np.random.RandomState(0)
model = MLPClassifier(solver='adam', alpha=0.001, learning_rate='constant', activation='relu',
                   hidden_layer_sizes=(32,32), random_state=1)


model.fit(x_train,y_train)
y_score = model.predict_proba(x_test)

# Compute ROC curve and ROC area for each class
fpr_mlp = dict()
tpr_mlp = dict()
roc_auc_mlp = dict()
for i in range(n_classes):
    fpr_mlp[i], tpr_mlp[i], _ = roc_curve(y3_mlp[:, i], y_score[:, i])
    roc_auc_mlp[i] = auc(fpr_mlp[i], tpr_mlp[i])
    print(roc_auc_mlp[i])

df_prob1 = pd.DataFrame({'MRN':df2_test.MRN,'filename':df2_test.filename, 'label':y_test})
d_pred2 = pd.DataFrame(y_score)
d_pred2.columns = ["chp","nsip","o","sar",'uip']
df_prob1 = pd.concat([df_prob1, d_pred2], axis=1)


def get_series_name(f):
    x = f[:-8]
    return x

d1=y_score
s1=np.argmax(d1,axis=1)
tru_label=y_val.to_numpy()
accuracy_score(tru_label,s1)
