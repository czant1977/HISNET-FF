import joblib
import os
from sklearn.ensemble import RandomForestRegressor
# 创建随机森林回归模型
#RFR = RandomForestRegressor()
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
#训练数据输入，并转换为特征向量
import ast
from sklearn.naive_bayes import MultinomialNB
import numpy as np
def scale_minmax(col,max,min):
    col = np.array(col)
    max = np.array(max)
    min = np.array(min)
    return (col-min)/(max-min).tolist()
class_num = '1'
shu_list = ['51']#['Scapanus','Mogera','Scaptonyx','Uropsilus','Euroscaptor','Parascaptor','Talpa','18']
feature_file = './feature_file/b7_'+class_num
if os.path.exists('./scaler_dir_b7'+class_num) is False:
        os.makedirs('./scaler_dir_b7'+class_num)
# ls = [teeth_max,teeth_min,head_max,head_min]
# print(len(ls))
for shu in shu_list:
    teeth_path = feature_file+'/'+shu+'/teeth_data'
    head_path = feature_file+'/'+shu+'/head_data'
    #head_path = './feature_file2/'+shu+'/teeth_data'
    model = MultinomialNB(alpha=1.0)
    #model = SVC()
    #model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    name = str(model).split('(')[0]
    print('正在读取数据。。。')
    for t,h in [(1,1)]:#[(1,9),(2,8),(3,7),(4,6),(6,4),(7,3),(8,2),(9,1)]:
        for pathnum,file_path in enumerate([teeth_path,head_path]):
            f1=open(os.path.join(file_path,'x_train.txt'),mode='r')
            f2=open(os.path.join(file_path,'y_train.txt'),mode='r')
            # if(pathnum==0):
            #     f3=open(os.path.join(file_path+'_od','x_test.txt'),mode='r')
            #     f4=open(os.path.join(file_path+'_od','y_test.txt'),mode='r')
            # else:
            f3=open(os.path.join(file_path,'x_test.txt'),mode='r')
            f4=open(os.path.join(file_path,'y_test.txt'),mode='r')
            x_train_list=f1.readlines()
            #print('zhe')
            y_train_list=f2.readlines()
            x_test_list=f3.readlines()
            #print(len(x_test_list[0]))
            y_test_list=f4.readlines()
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            for ls in [x_train_list,y_train_list,x_test_list,y_test_list]:
                for i in range(len(ls)):
                    ls[i]=eval(ls[i].strip())
            
            #print(type(x_train_list))
            concat_lei = 'concat_'+str(t)+':'+str(h)+'_data'
            if(pathnum==0):
                print('teeth')
                scaler = MinMaxScaler(feature_range=(0,t))#牙齿占比
                scaler.fit(x_train_list)
                x_train_list_teeth = torch.tensor(scaler.transform(x_train_list))
                x_test_list_teeth = torch.tensor(scaler.transform(x_test_list))
                joblib.dump(scaler,os.path.join('./scaler_dir_b7'+class_num,shu+'_teeth.pkl'))
            else: 
                print('head')
                scaler = MinMaxScaler(feature_range=(0,h))
                scaler.fit(x_train_list)
                x_train_list_head = torch.tensor(scaler.transform(x_train_list))
                x_test_list_head = torch.tensor(scaler.transform(x_test_list))
                joblib.dump(scaler,os.path.join('./scaler_dir_b7'+class_num,shu+'_head.pkl'))

        #shu = shu+'_2ceng'
        x_train = torch.concat((x_train_list_teeth,x_train_list_head),dim=1).tolist()
        y_train = y_train_list
        x_test = torch.concat((x_test_list_teeth,x_test_list_head),dim=1).tolist()
        y_test = y_test_list
        if os.path.exists(os.path.join(feature_file,shu,concat_lei)) is False:
            os.makedirs(os.path.join(feature_file,shu,concat_lei))
        f1=open(os.path.join(feature_file,shu,concat_lei,'x_train.txt'),mode='w')
        f2=open(os.path.join(feature_file,shu,concat_lei,'y_train.txt'),mode='w')
        f3=open(os.path.join(feature_file,shu,concat_lei,'x_test.txt'),mode='w')
        f4=open(os.path.join(feature_file,shu,concat_lei,'y_test.txt'),mode='w')
        f1.writelines('\n'.join(map(str, x_train)))
        f2.writelines('\n'.join(map(str, y_train)))
        f3.writelines('\n'.join(map(str, x_test)))
        f4.writelines('\n'.join(map(str, y_test)))
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        print(len(x_train))
        print(len(x_train[0]))
        print(len(y_train))
        print('正在训练。。。')
        model.fit(x_train,y_train)
        print('正在测试。。。')
        train_acc = model.score(x_train,y_train)
        test_acc = model.score(x_test,y_test)
        print('{}属训练集准确率:{}\n测试集准确率:{}'.format(shu,train_acc,test_acc))
        joblib.dump(model, os.path.join(feature_file,shu,concat_lei,name+'_'+str(test_acc)+'.pkl'))
