import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
def out_feature(teeth_img,head_img,numclass,torh,shu,class_num):
    model_tooth = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_tooth._fc.in_features
    model_tooth._fc = nn.Linear(num_ftrs, numclass)
    tooth_net = "/home/cz/data/final_data/teeth_5zhe_data/teeth_data_"+class_num+"/weights-b7-teeth/EfficientNetb7-teeth_50_"+str(numclass)+"_"+shu+"/best_network_eb4.pth"
    model_tooth.load_state_dict(torch.load(tooth_net, map_location=device))
    model_tooth = model_tooth.to(device)
    model_tooth.eval()
    for param in model_tooth.parameters():
        param.requires_grad = False
    model_whole = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_whole._fc.in_features
    model_whole._fc = nn.Linear(num_ftrs, numclass)
    whole_net = "/home/cz/data/final_data/head_5zhe_data/head_data_"+class_num+"/weights-b7-head/EfficientNetb7-head_50_"+str(numclass)+"_"+shu+"/best_network_eb4.pth"
    model_whole.load_state_dict(torch.load(whole_net, map_location=device))
    model_whole = model_whole.to(device)
    model_whole.eval()
    for param in model_whole.parameters():
        param.requires_grad = False
    with torch.no_grad():
        teeth_img = teeth_img.to(device)
        head_img = head_img.to(device)
        if(torh == 'teeth'):
            #rint('teeth')
            tooth = model_tooth.extract_endpoints(teeth_img)
            return tooth['reduction_6']
        else:
            #print('head')
            whole = model_whole.extract_endpoints(head_img)
            return whole['reduction_6']
def out_feature_18(teeth_img,head_img,torh,class_num):
    model_tooth = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_tooth._fc.in_features
    model_tooth._fc = nn.Linear(num_ftrs, 18)
    tooth_net = "/home/cz/data/final_data/teeth_5zhe_data/teeth_data_"+class_num+"/weights-b7-teeth/EfficientNet_50_18/best_network_eb4.pth"
    model_tooth.load_state_dict(torch.load(tooth_net, map_location=device))
    model_tooth = model_tooth.to(device)
    model_tooth.eval()
    for param in model_tooth.parameters():
        param.requires_grad = False
    model_whole = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_whole._fc.in_features
    model_whole._fc = nn.Linear(num_ftrs, 18)
    whole_net = "/home/cz/data/final_data/head_5zhe_data/head_data_"+class_num+"/weights-b7-head/EfficientNet_50_18/best_network_eb4.pth"
    model_whole.load_state_dict(torch.load(whole_net, map_location=device))
    model_whole = model_whole.to(device)
    model_whole.eval()
    for param in model_whole.parameters():
        param.requires_grad = False
    with torch.no_grad():
        teeth_img = teeth_img.to(device)
        head_img = head_img.to(device)
        if(torh == 'teeth'):
            #rint('teeth')
            tooth = model_tooth.extract_endpoints(teeth_img)
            return tooth['reduction_6']
        else:
            #print('head')
            whole = model_whole.extract_endpoints(head_img)
            return whole['reduction_6']
def out_feature_51(teeth_img,head_img,torh,class_num):
    model_tooth = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_tooth._fc.in_features
    model_tooth._fc = nn.Linear(num_ftrs, 51)
    tooth_net = "/home/cz/data/final_data/teeth_5zhe_data/teeth_data_"+class_num+"/weights-b7-teeth/EfficientNet_50_51/best_network_eb4.pth"
    model_tooth.load_state_dict(torch.load(tooth_net, map_location=device))
    model_tooth = model_tooth.to(device)
    model_tooth.eval()
    for param in model_tooth.parameters():
        param.requires_grad = False
    model_whole = EfficientNet.from_name('efficientnet-b7').to(device)
    num_ftrs = model_whole._fc.in_features
    model_whole._fc = nn.Linear(num_ftrs, 51)
    whole_net = "/home/cz/data/final_data/head_5zhe_data/head_data_"+class_num+"/weights-b7-head/EfficientNet_50_51/best_network_eb4.pth"
    model_whole.load_state_dict(torch.load(whole_net, map_location=device))
    model_whole = model_whole.to(device)
    model_whole.eval()
    for param in model_whole.parameters():
        param.requires_grad = False
    with torch.no_grad():
        teeth_img = teeth_img.to(device)
        head_img = head_img.to(device)
        if(torh == 'teeth'):
            #rint('teeth')
            tooth = model_tooth.extract_endpoints(teeth_img)
            return tooth['reduction_6']
        else:
            #print('head')
            whole = model_whole.extract_endpoints(head_img)
            return whole['reduction_6']
def image_process(ipath,transform,input_size):
    image = Image.open(ipath)
    image = image.convert('RGB')
    image = transform(image)
    image = torch.reshape(image, (1, 3, input_size, input_size))
    image = image.to(device)
    return image
def guiyihua(teeth_feature,head_feature,scaler_teeth,scaler_head):
    teeth_feature = teeth_feature.cpu().detach().numpy().tolist()
    head_feature = head_feature.cpu().detach().numpy().tolist()
    teeth_feature = scaler_teeth.transform(teeth_feature)
    teeth_feature = torch.tensor(teeth_feature,dtype=torch.float)
    head_feature = scaler_head.transform(head_feature)
    head_feature = torch.tensor(head_feature,dtype=torch.float)
        #f2=open(os.path.join(file_path,'x_train.txt'),mode='r')
    concat_feature = torch.concat((teeth_feature,head_feature),dim=1).to(device)
    return concat_feature
def scale_minmax(col,max,min):
    col = np.array(col)
    max = np.array(max)
    min = np.array(min)
    return (col-min)/(max-min).tolist()

def guiyihua2(teeth_feature,head_feature,teeth_transform,head_transform):
    teeth_feature = teeth_feature.cpu().detach().numpy()#.tolist()
    head_feature = head_feature.cpu().detach().numpy()#.tolist()
    teeth_feature = teeth_feature/np.array(teeth_transform)
    teeth_feature = torch.tensor(teeth_feature,dtype=torch.float)
    head_feature = head_feature/np.array(head_transform)/25.0
    head_feature = torch.tensor(head_feature,dtype=torch.float)
    concat_feature = torch.concat((teeth_feature,head_feature),dim=1).to(device)
    return concat_feature