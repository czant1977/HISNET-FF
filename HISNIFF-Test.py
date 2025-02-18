import torch
from torch import nn
import torchvision
from torchvision import transforms
import os
from feature_file.outfeature import out_feature,out_feature_18,image_process,guiyihua,guiyihua2
from liner_model import SimpleNet2,SimpleNet
from feature_file.GYH import scaler_tool
def reload_tools():
    print('正在读取归一化工具')
    model_list = ['Uropsilus', 'Talpa', 'Scaptonyx', 'Scapanus', 'Parascaptor', 'Mogera', 'Euroscaptor']
    shu_tool = []
    for shu in model_list:
        teeth_tool,head_tool = scaler_tool(shu)
        shu_tool.append([teeth_tool,head_tool])
    teeth_tool, head_tool = scaler_tool('18')
    shu_tool.append([teeth_tool, head_tool])
    return shu_tool
def test_combine(teeth_img_path,head_img_path,shu_tool):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model_list = ['Uropsilus', 'Talpa', 'Scaptonyx', 'Scapanus', 'Parascaptor', 'Mogera', 'Euroscaptor']
    num_list = [5, 5, 4, 4, 3, 9, 8]
    class_18 = ['Condylura', 'Desmana', 'Dymecodon', 'Euroscaptor', 'Galemys', 'Mogera',
                'Neurotrichus', 'Orescaptor', 'Parascalops', 'Parascaptor', 'Scalopus',
                'Scapanulus', 'Scapanus', 'Scaptochirus', 'Scaptonyx', 'Talpa', 'Uropsilus', 'Urotrichus']
    zhong_model_list = [['Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes'],
                        ['Talpa altaica', 'Talpa caeca', 'Talpa davidiana', 'Talpa europaea', 'Talpa romana'],
                        ['Scaptonyx Scaptonyx sp1', 'Scaptonyx Scaptonyx sp2', 'Scaptonyx Scaptonyx sp4', 'Scaptonyx Scaptonyx sp5'],
                        ['Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii'],
                        ['Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                        ['Mogera etigo', 'Mogera hainana','Mogera imaizumii', 'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta','Mogera tokudae', 'Mogera wogura'],
                        ['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens']]
    input_size=480
    transform = torchvision.transforms.Compose(
        [
                transforms.Resize((input_size,input_size)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    process = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1),
        nn.ReLU(inplace=True),
    )
    process.to(device)
    net_18 = torch.load('./feature_file/weights-al-5epoch/18/concat_5:1/best_model.pkl')
    net_18.to(device)
    net_18.eval()
    head_image = image_process(head_img_path,transform,input_size)
    teeth_image = image_process(teeth_img_path,transform,input_size)
    head_feature_shu = out_feature_18(teeth_image,head_image,'head')
    teeth_feature_shu = out_feature_18(teeth_image,head_image,'teeth')
    with torch.no_grad():
        head_feature_shu = process(head_feature_shu)
        teeth_feature_shu = process(teeth_feature_shu)
        concat_feature_shu = guiyihua(teeth_feature_shu,head_feature_shu,shu_tool[-1][0],shu_tool[-1][1])
        shu_output = net_18(concat_feature_shu)
        shu_result = shu_output.argmax(1)
        shu_result = str(shu_result).split('[')[1].split(']')[0]
        shu_result = class_18[int(shu_result)]
        if(shu_result not in model_list):
            zhong_result = shu_result
            return shu_result
        else:
            shu_id = model_list.index(shu_result)
            num_class = num_list[shu_id]
            head_feature_shu = out_feature(teeth_image,head_image,num_class,'head',shu_result)
            teeth_feature_shu = out_feature(teeth_image,head_image,num_class,'teeth',shu_result)
            head_feature_shu = process(head_feature_shu)
            teeth_feature_shu = process(teeth_feature_shu)
            concat_feature_shu = guiyihua(teeth_feature_shu,head_feature_shu,shu_tool[shu_id][0],shu_tool[shu_id][1])
            net_49 = torch.load(os.path.join('./feature_file/weights-al-5epoch',shu_result,'concat_5:1','best_model.pkl'),map_location=device)
            net_49.eval()
            zhong_output = net_49(concat_feature_shu)
            zhong_result = zhong_output.argmax(1)
            zhong_result = str(zhong_result).split('[')[1].split(']')[0]
            zhong_result = zhong_model_list[shu_id][int(zhong_result)]
            return zhong_result
shu_tool = reload_tools()
head_path = '/home/cz/data/feature-mouse/data_al/head_al_49_1/test/Mogera etigo/Mogera#etigo#13204M#Img2090#s#v.JPG'
teeth_path = '/home/cz/data/feature-mouse/data_al/teeth_al_od/teeth_al_49_od/test/Mogera etigo/Mogera#etigo#13204M#Img2090#s#v.JPG'
print(test_combine(teeth_path,head_path,shu_tool))

