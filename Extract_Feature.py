import os
from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
#from torch._jit_internal import weak_module, weak_script_method
from torch.nn.parameter import Parameter
import math
from PIL import Image
from torchvision import models,transforms
import torchvision
from sklearn.svm import LinearSVC,SVC
#from vgg_model import VGG16_C3
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from sklearn.naive_bayes import MultinomialNB,GaussianNB
#from mlp_model import out_feature
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
batch_size = 32
input_size = 600
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global dropout_rate
print(device)
def out_feature(teeth_img,head_img,numclass,num,torh):
    #model_tooth = torchvision.models.efficientnet_b4().to(device)
    model_tooth = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_tooth._fc.in_features
    model_tooth._fc = nn.Linear(num_ftrs, numclass)
    tooth_net = "../teeth_5zhe_data/teeth_data_"+class_num+"/weights-b7-teeth/EfficientNetb7-teeth_50_"+str(num)+"_"+shu+"/best_network_eb4.pth"
    model_tooth.load_state_dict(torch.load(tooth_net, map_location=device))
    model_tooth = model_tooth.to(device)
    model_tooth.eval()
    for param in model_tooth.parameters():
        param.requires_grad = False
    #model_whole = torchvision.models.efficientnet_b4().to(device)
    model_whole = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_whole._fc.in_features
    model_whole._fc = nn.Linear(num_ftrs, numclass)
    whole_net = "../head_5zhe_data/head_data_"+class_num+"/weights-b7-head/EfficientNetb7-head_50_"+str(num)+"_"+shu+"/best_network_eb4.pth"
    model_whole.load_state_dict(torch.load(whole_net, map_location=device))
    model_whole = model_whole.to(device)
    model_whole.eval()
    for param in model_whole.parameters():
        param.requires_grad = False
    with torch.no_grad():
        teeth_img = teeth_img.to(device)
        head_img = head_img.to(device)
        
        
        #tooth = normal(tooth)
        #whole = normal(whole)
        #x = torch.concat((tooth, whole), dim=1)
        if(torh == 'teeth'):
            #rint('teeth')
            tooth = model_tooth.extract_endpoints(teeth_img)
            return tooth['reduction_6']
        else:
            #print('head')
            whole = model_whole.extract_endpoints(head_img)
            return whole['reduction_6']
# 指定训练所用的GPU编号0-9
# def signed_sqrt_normalization(data):
#     sign = torch.sign(data)
#     sqrt_abs = torch.sqrt(torch.abs(data))
#     normalized_data = sign * sqrt_abs
#     return normalized_data
def transform_change(epoch):
    if(epoch==0):
        transform = transforms.Compose(
                                            [
                                                transforms.Resize((input_size, input_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ]
                                        )
    elif(0<epoch<3):
        transform = transforms.Compose(
                                        [
                                            transforms.Resize((input_size, input_size)),
                                            #transforms.RandomResizedCrop(input_size),
                                            transforms.RandomHorizontalFlip(),
                                            #transforms.CenterCrop(input_size),
                                            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                            #transforms.ColorJitter(contrast=2),
                                            #transforms.ColorJitter(saturation=20),
                                            #transforms.ColorJitter(hue=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ]
                                    )
    else:
        transform = transforms.Compose(
                                        [
                                            transforms.Resize((input_size, input_size)),
                                            #transforms.RandomResizedCrop(input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(input_size),
                                            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                            transforms.ColorJitter(contrast=2),
                                            transforms.ColorJitter(saturation=20),
                                            #transforms.ColorJitter(hue=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ]
                                    )
    return transform
test_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        #transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
process_test = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    #nn.MaxPool2d(kernel_size=3, stride=3,padding=1),
    #nn.MaxPool2d(kernel_size=3, stride=3),
    nn.Flatten(start_dim=1),
    #nn.Dropout(dropout_rate),
    #nn.Linear(2048,2048),
    nn.ReLU(inplace=True),
)
process_test.to(device)
process_train = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    #nn.MaxPool2d(kernel_size=3, stride=3,padding=1),
    #nn.MaxPool2d(kernel_size=3, stride=3),
    nn.Flatten(start_dim=1),
    #nn.Linear(2048,2048),
    nn.ReLU(inplace=True),
)
process_train.to(device)
#process.eval()
shu_list = ['Scapanus','Mogera','Scaptonyx','Uropsilus','Euroscaptor','Parascaptor','Talpa']
zhong_list = [4,9,3,5,8,2,7]
# 创建朴素贝叶斯分类器
model = MultinomialNB(alpha=1.0)
#model = SVC()
epochs= 1
class_num = '3'
server_name = 'b7_'+class_num
for zhong,shu in enumerate(shu_list):
    print(shu)
    for torh in ['teeth','head']:#['teeth','head']:#分别提取头骨牙齿信息
        best_acc=0
        print(torh)
        process = process_test
        process.eval()
        file_path = './feature_file/'+server_name+'/'+shu+'/'+torh+'_data'
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        print(file_path.split('/')[-1].split('_')[0])
        datapath = '../head_5zhe_data/head_data_'+class_num+'/head_2level_r_p/'+shu
        teeth_datapath = '../teeth_5zhe_data/teeth_data_'+class_num+'/teeth_2level_r_p/'+shu
        feature_test_list = []
        y_test_list = []
        ty_test_list = []
        test_data = torchvision.datasets.ImageFolder(root=os.path.join(datapath, "test"), transform=test_transform)
        test_dataloader = DataLoader(test_data,batch_size = batch_size, shuffle=False,num_workers=8)
        teeth_test_data = torchvision.datasets.ImageFolder(root=os.path.join(teeth_datapath, "test"), transform=test_transform)
        teeth_test_dataloader = DataLoader(teeth_test_data,batch_size = batch_size, shuffle=False,num_workers=8)
        num_class = len(os.listdir(os.path.join(datapath,'test')))
        with tqdm(zip(test_dataloader,teeth_test_dataloader),desc="正在提取测试集特征",leave=False) as test_bar:
            for(x_test,y_test),(tx_test,ty_test) in test_bar:
                x_test = x_test.to(device)
                y_test = y_test.numpy()
                tx_test = tx_test.to(device)
                ty_test = ty_test.numpy()
                feature_test = out_feature(tx_test,x_test,num_class,zhong_list[zhong],torh)
                with torch.no_grad():
                    feature_test = process(feature_test)
                #feature_test = signed_sqrt_normalization(feature_test)
                feature_test = feature_test.cpu().detach().numpy()
                for i in range(len(feature_test)):
                    feature_test_list.append(str(feature_test[i].tolist()))
                    y_test_list.append(str(y_test[i]))
                    ty_test_list.append(str(ty_test[i]))
            f1 = open(file_path+'/x_test.txt',mode='w')
            f2 = open(file_path+'/y_test.txt',mode='w')
            f3 = open(file_path+'/ty_test.txt',mode='w')
            f1.write('\n'.join(feature_test_list))
            f2.write('\n'.join(y_test_list))
            f3.write('\n'.join(ty_test_list))
            f1.close()
            f2.close()
            f3.close()
            #print(feature_test_list)
            print(y_test_list[:50])
            print(ty_test_list[:50])
            feature_train_list = []
            y_train_list = []
            ty_train_list = []
        for epoch in range(epochs):#后期数据增强修改轮数
            transform = transform_change(epoch)
            train_data = torchvision.datasets.ImageFolder(root=os.path.join(datapath, "train"), transform=transform)
            # shuffle:是否打乱数据集 num_workers:cpu数量
            train_dataloader = DataLoader(train_data,batch_size = batch_size , shuffle=False,num_workers=8)
            teeth_train_data = torchvision.datasets.ImageFolder(root=os.path.join(teeth_datapath, "train"), transform=transform)
            # shuffle:是否打乱数据集 num_workers:cpu数量
            teeth_train_dataloader = DataLoader(teeth_train_data,batch_size=batch_size, shuffle=False,num_workers=8)


            print(train_data.class_to_idx)
            num_class = len(os.listdir(os.path.join(datapath,'train')))
            train_data_size = len(train_data)
            test_data_size = len(test_data)
            print("训练集长度{}".format(train_data_size))
            print("测试集长度{}".format(test_data_size))


            with tqdm(zip(train_dataloader,teeth_train_dataloader),desc="第{}轮，正在提取训练集特征".format(epoch)) as train_bar:
                for(x_train,y_train),(tx_train,ty_train) in train_bar:
                    x_train = x_train.to(device)
                    tx_train = tx_train.to(device)
                    y_train = y_train.numpy()
                    ty_train = ty_train.numpy()
                    feature_train = out_feature(tx_train,x_train,num_class,zhong_list[zhong],torh)
                    #print(feature_train.shape)
                    feature_train = process(feature_train)
                    #print(feature_train.shape)
                    #feature_train = signed_sqrt_normalization(feature_train)
                    feature_train = feature_train.cpu().detach().numpy()
                    #print(feature_train)
                    #print(y_train)
                    for i in range(len(feature_train)):
                        feature_train_list.append(str(feature_train[i].tolist()))
                        y_train_list.append(str(y_train[i]))
                        ty_train_list.append(str(ty_train[i]))
        f1 = open(file_path+'/x_train.txt',mode='w')
        f2 = open(file_path+'/y_train.txt',mode='w')
        f1.write('\n'.join(feature_train_list))#写入训练集特征，每张图片对应一个2048的序列
        f2.write('\n'.join(y_train_list))#训练集类别特征，0,1,2,3,4....
        f1.close()
        f2.close()
            #print(feature_train_list)
            #print(feature_train_list[:5])
            #print(ty_train_list[:5])
        #model_path = './feature_file/'+shu+'/concat_data'
        f1=open(os.path.join(file_path,'x_train.txt'),mode='r')
        f2=open(os.path.join(file_path,'y_train.txt'),mode='r')
        f3=open(os.path.join(file_path,'x_test.txt'),mode='r')
        f4=open(os.path.join(file_path,'y_test.txt'),mode='r')
        x_train_list=f1.readlines()
        #print('zhe')
        y_train_list=f2.readlines()
        x_test_list=f3.readlines()
        #print(len(x_test_list[0]))
        y_test_list=f4.readlines()
        for ls in [x_train_list,y_train_list,x_test_list,y_test_list]:
            for i in range(len(ls)):
                ls[i]=eval(ls[i].strip())
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        
        name = str(model)[:-2]
        scaler = MinMaxScaler(feature_range=(0,1))#归一化
        scaler.fit(x_train_list)
        x_train_list = scaler.transform(x_train_list)
        x_test_list = scaler.transform(x_test_list)
        print(len(x_train_list[0]))
        model.fit(x_train_list,y_train_list)#送入朴素贝叶斯分类器训练
        #model = joblib.load("./feature_file/head_data/SVC_0.9230769230769231.pkl")
        print('正在测试。。。')
        train_acc = model.score(x_train_list,y_train_list)#测试
        test_acc = model.score(x_test_list,y_test_list)
        print('{}属训练集准确率:{}\n测试集准确率:{}'.format(shu,train_acc,test_acc))
        # if(test_acc>best_acc):
        #     best_acc=test_acc
        joblib.dump(model, os.path.join(file_path,name+'_'+str(test_acc)+'.pkl'))#保存模型
        # if(test_acc>=0.90):
        #     joblib.dump(model, os.path.join(file_path,name+'_'+str(test_acc)+'.pkl'))
        #     break