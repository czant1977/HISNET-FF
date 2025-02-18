import torch
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
def scaler_tool(shu):
    feature_file = './weight-combine/al-5epoch'
    #concat_lei = 'concat_5:1_data'
    teeth_path = feature_file+'/'+shu+'/teeth_data'
    head_path = feature_file+'/'+shu+'/head_data'
    for pathnum,file_path in enumerate([teeth_path,head_path]):
        f=open(os.path.join(file_path,'x_train.txt'),mode='r')
        x_train_list=f.readlines()
        f.close()
        for i in range(len(x_train_list)):
            x_train_list[i]=eval(x_train_list[i].strip())
        if(pathnum==0):
            #print('teeth')
            scaler_teeth = MinMaxScaler(feature_range=(0,1))#牙齿占比
            scaler_teeth.fit(x_train_list)
        else:
            #print('head')
            scaler_head = MinMaxScaler(feature_range=(0,0.2))#头骨占比
            scaler_head.fit(x_train_list)
    return scaler_teeth,scaler_head
