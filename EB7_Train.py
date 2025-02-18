import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils import tensorboard
import time
from Early_Stopping import EarlyStopping
import os 
from torch import nn
from matplotlib import pyplot as plt
from torchvision.models import EfficientNet_B5_Weights
from efficientnet_pytorch import EfficientNet
import torch.cuda.amp as amp
from wepub import SendMessage
from torchvision import models
sm = SendMessage()
def test_net(net,test_dataloader,loss_func):
    net.eval()
    eval_loss1 = 0
    eval_acc1 = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_func(outputs, targets)
            eval_loss1 += loss.item() * targets.size(0)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == targets).sum()
            eval_acc1 += num_correct.item()

        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss1 / test_data_size
                                                      , eval_acc1 / test_data_size))
    return eval_acc1/test_data_size
#torch.cuda.set_device(1) 
if __name__ == '__main__':
    '''
        参数
    '''
    loss_list = []
    acc_list = []
    t_acc_list = []
    t_loss_list = []
    # 每轮训练喂入数据量的大小
    batch_size = 4
    # 初始学习率
    learning_rate = 0.01
    # 总的训练轮数
    num_epochs = 50
    # 学习率迭代的间隔
    step_size = 10
    # 学习率迭代的间隔
    #num_class = 18
    # EarlyStop等待的轮数
    patience = 50
    h_t = 'head'
    class_num=1
    #datapath = './'+h_t+'_data/'+h_t+'_18_r_p_rand'
    datapath = './'+h_t+'_5zhe_data/'+h_t+'_data_'+str(class_num)+'/'+h_t+'_51_r_p_aug'
    #datapath = './teeth_5zhe_data/teeth_data_1/teeth_2level_r_p/Mogera'
    server_name = 'b7-'+h_t
    input_size = 600
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    print(device)
    # 指定训练所用的GPU编号0-9
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            #transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(input_size),
            # # transforms.RandomRotation(90),
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # transforms.ColorJitter(contrast=2),
            # transforms.ColorJitter(saturation=20),
            #transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            #transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    train_data = torchvision.datasets.ImageFolder(root=os.path.join(datapath, "train"), transform=transform)
    # shuffle:是否打乱数据集 num_workers:cpu数量
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)

    test_data = torchvision.datasets.ImageFolder(root=os.path.join(datapath, "test"), transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=8)

    print(train_data.class_to_idx)
    num_class = len(os.listdir(os.path.join(datapath,'train')))
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集长度{}".format(train_data_size))
    print("测试集长度{}".format(test_data_size))
    net = EfficientNet.from_pretrained('efficientnet-b7').to(device)
    # num_ftrs = net._fc.in_features
    # net._fc = nn.Linear(num_ftrs, 18)
    #net.load_state_dict(torch.load("/home/cz/data/final_data/weights-b5-teeth_r_p/adv-efficientnet-b5-86493f6b.pth",map_location = device))
    num_ftrs = net._fc.in_features
    net._fc = nn.Linear(num_ftrs, num_class)
    # net.load_state_dict(torch.load("/home/cz/data/final_data/teeth_5zhe_data/teeth_data_1/weights-b5-teeth/EfficientNetb5-teeth_100_9_Mogera_aug/best_network_eb4.pth"))
    #net = torchvision.models.vgg16(pretrained=True)
    
    #net.load_state_dict(torch.load("/data_16t/zengweiqi/two_level/weights-b5-2/EfficientNetB5_100_4_Scaptonyx/best_network_eb4.pth"))
    #{'Condylura': 0, 'Desmana': 1, 'Dymecodon': 2, 'Euroscaptor': 3, 'Galemys': 4, 'Mogera': 5, 'Neurotrichus': 6, 'Orescaptor': 7, 'Parascalops': 8, 'Parascaptor': 9, 'Scalopus': 10, 'Scapanulus': 11, 'Scapanus': 12, 'Scaptochirus': 13, 'Scaptonyx': 14, 'Talpa': 15, 'Uropsilus': 16, 'Urotrichus': 17}
    #net = torchvision.models.efficientnet_b5(weights=("pretrained",EfficientNet_B5_Weights.IMAGENET1K_V1))
    #{'Condylura': 0, 'Desmana': 1, 'Dymecodon': 2, 'Euroscaptor': 3, 'Galemys': 4, 'Mogera': 5, 'Neurotrichus': 6, 'Orescaptor': 7, 'Parascalops': 8, 'Parascaptor': 9, 'Scalopus': 10, 'Scapanulus': 11, 'Scapanus': 12, 'Scaptochirus': 13, 'Scaptonyx': 14, 'Talpa': 15, 'Uropsilus': 16, 'Urotrichus': 17}
    #net = torchvision.models.AlexNet(num_classes=18)
    # 修改最后一层全连接层
    #net.classifier[1] = torch.nn.Linear(1536, num_class)
    #net.classifier[1] = torch.nn.Linear(2048, num_class)
    
    #print(num_ftrs)
    #net.classifier[1] = nn.Linear(2048, num_class)
    #net.load_state_dict(torch.load("./weights-b5-2-head/EfficientNet_100_18head/best_network_eb4.pth"))#迁移学习or继续学习
    #net = torchvision.models.vit_l_16(pretrained=True)
    net = net.to(device)
    net_class = str(net).split('(')[0]
    name_style = net_class+"_"+str(num_epochs)+"_"+str(num_class)+"_aug"
    save_path = './'+h_t+'_5zhe_data/'+h_t+'_data_'+str(class_num)+"/weights-"+server_name+"/"+name_style
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    earlt_stopping = EarlyStopping(save_path, patience, verbose=True)
    #et.load_state_dict(torch.load("./weights/" + net_class + "_" + str(num_epochs) + "/bestwork.pth"))
    #print(net)
    #设置损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)
    # net._dropout = nn.Dropout(0.7)
    #设置优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #设置学习率迭代器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8, last_epoch=-1)
    # ********************************训练日志保存位置******************************* #
    writer = tensorboard.SummaryWriter(save_path+'/logs')
    best_acc = test_net(net,test_dataloader,loss_func)
    earlt_stopping(best_acc, net)
    best_epoch = 0

    start = time.time()
    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        running_acc = 0
        print("——————————————————————————————{}——————————————————————————————".format(epoch + 1))
        print("lr = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        #for data in tqdm(enumerate(train_dataloader)):
        with tqdm(train_dataloader, colour='green', leave=False) as t:
            for data in t:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                with amp.autocast(enabled=True):
                    outputs = net.forward(imgs)
                    loss = loss_func(outputs, targets)

                total_train_loss += loss.item() * targets.size(0)
                _, pred = torch.max(outputs, 1)  # 预测最大值所在的位置标签
                num_correct = (pred == targets).sum()
                accuracy = (pred == targets).float().mean()
                running_acc += num_correct.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, total_train_loss / train_data_size, running_acc / train_data_size))
        loss_list.append(total_train_loss/train_data_size)
        acc_list.append(running_acc/train_data_size)
        writer.add_scalar("train_loss", total_train_loss / train_data_size, epoch)
        writer.add_scalar("train_acc", running_acc / train_data_size, epoch)
        scheduler.step()

        # 测试开始
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                #with amp.autocast(enabled=True):
                outputs = net(imgs)
                loss = loss_func(outputs, targets)
                eval_loss += loss.item() * targets.size(0)
                _, pred = torch.max(outputs, 1)
                num_correct = (pred == targets).sum()
                eval_acc += num_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_size
                                                          , eval_acc / test_data_size))
            if eval_acc / test_data_size >= best_acc:
                best_acc = eval_acc / test_data_size
                best_epoch=epoch
                f = open(save_path+'/best_acc.txt',mode = 'w')
                f.write(str(train_data.class_to_idx)+'\n')
                f.write("训练集长度{}\n".format(train_data_size))
                f.write("测试集长度{}\n".format(test_data_size))
                f.write('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}\n'.format(
            epoch + 1, total_train_loss / train_data_size, running_acc / train_data_size))
                f.write('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_size
                                                        , eval_acc / test_data_size))
                f.close()
            t_loss_list.append(eval_loss / test_data_size)
            t_acc_list.append(eval_acc / test_data_size)
            writer.add_scalar("test_loss", eval_loss / test_data_size, epoch)
            writer.add_scalar("test_acc", eval_acc / test_data_size, epoch)
            writer.add_scalar("learning_rate", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            print()
        valid_acc = eval_acc / test_data_size
        earlt_stopping(valid_acc, net)
        if earlt_stopping.early_stop:
            print("Early Stopping")
            # ******************************** 模型文件保存位置 ******************************* 
            #torch.save(net.state_dict(), "./weights/Eb3_{}_low_p.pth".format(epoch + 1))
            print("Best acc model is saved!")
            break
    end = time.time()
    best_epoch = epoch+1 - earlt_stopping.counter
    print("Finish all epochs, best epoch: {}, best acc: {}".format(best_epoch, best_acc))
    print("running time:", end - start)
    json_data = {
            "shu": server_name,
            "train_acc": running_acc / train_data_size,
            "test_acc": best_acc,
            "epoch": best_epoch,
            "second": end-start
        }
        # 发送消息
    sm.send_message(json_data=json_data)
    epochs_list = [i for i in range(epoch+1)]
    plt.subplot(2, 2, 1)
    plt.title("loss line")
    plt.plot(loss_list)
    plt.subplot(2, 2, 2)
    plt.title("acc line")
    plt.plot(acc_list)
    plt.subplot(2, 2, 3)
    plt.title("test_loss line")
    plt.plot(t_loss_list)
    plt.subplot(2, 2, 4)
    plt.title("test_acc line")
    plt.plot(t_acc_list)
    plt.savefig(save_path+'/process_img.png')
    plt.show()
    writer.close()
