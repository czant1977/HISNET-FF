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
from torchvision.models import EfficientNet_B5_Weights,EfficientNet_V2_L_Weights
from efficientnet_pytorch import EfficientNet
import torch.cuda.amp as amp
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
lei = 'teeth'
class_num = 3
class_path = './'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_2level_r_p'
class_list = os.listdir(class_path)
model_list = ['Scapanus','Talpa','Parascaptor','Uropsilus','Scaptonyx','Mogera','Euroscaptor']
for classname in model_list:#到属
    num_class = len(os.listdir(os.path.join(class_path,classname,'train')))
    #print(num_class)
    if(num_class==1):
        print("{}属只有一个种，不做训练".format(classname))
        continue
    
    '''
        参数
    '''
    #net = EfficientNet.from_pretrained('efficientnet-b5')
    # net = EfficientNet.from_name('efficientnet-b5')
    #net = torchvision.models.efficientnet_v2_l(weights=("pretrained",EfficientNet_V2_L_Weights.DEFAULT))
    #num_ftrs = net._fc.in_features
    #net._fc = nn.Linear(num_ftrs, 18)
    #net.classifier[1] = torch.nn.Linear(1280, 18)
    
    #net.load_state_dict(torch.load("bestwork-b5.pth"))#迁移学习
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
    #num_class = 47
    # EarlyStop等待的轮数
    patience = 50

    input_size = 600
    server_name = 'b7-'+lei
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 指定训练所用的GPU编号0-9
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            #transforms.RandomResizedCrop((224,224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # transforms.CenterCrop(input_size),
            # transforms.ColorJitter(contrast=2),
            # transforms.ColorJitter(saturation=20),
            # transforms.ColorJitter(hue=0.2),
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
    train_data = torchvision.datasets.ImageFolder(root=os.path.join(class_path,classname,"train"), transform=transform)
    # shuffle:是否打乱数据集 num_workers:cpu数量
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=os.path.join(class_path,classname,"test"), transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(train_data.class_to_idx)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集长度{}".format(train_data_size))
    print("测试集长度{}".format(test_data_size))
    net = EfficientNet.from_name('efficientnet-b7').to(device)

    #net = torchvision.models.efficientnet_b5(weights=("pretrained",EfficientNet_B5_Weights.IMAGENET1K_V1))
    #net = torchvision.models.efficientnet_b4()
    #net = torchvision.models.AlexNet(num_0000000000000000000000000000000000  .classes=18)
    # 修改最后一层全连接层
    #net.classifier[1] = torch.nn.Linear(2048, num_class)
    
    
    
    
    net_class = str(net).split('(')[0]
    name_style = net_class+server_name+"_"+str(num_epochs)+"_"+str(num_class)+'_'+classname#命名格式
    server_name = 'b7-'+lei
    save_path = './'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+"/weights-"+server_name+"/"+name_style
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    #num_ftrs = net._fc.in_features
    #net._fc = nn.Linear(num_ftrs, num_class)
    #net.classifier[1] = nn.Linear(1792, num_class)
    #net = torchvision.models.vit_l_16(pretrained=True)
    #net = nn.DataParallel(net,device_ids = [1, 5])
    #net.classifier[1] = torch.nn.Linear(1280, num_class)
    
    
    try:
        num_ftrs = net._fc.in_features
        net._fc = nn.Linear(num_ftrs, num_class)
        net.load_state_dict(torch.load(os.path.join(save_path, 'best2_network_eb4.pth')))
    except:
        num_ftrs = net._fc.in_features
        net._fc = nn.Linear(num_ftrs, 18)
        
        net.load_state_dict(torch.load('./'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+"/weights-"+server_name+'/EfficientNet_50_18/best_network_eb4.pth',map_location = device))
                
        num_ftrs = net._fc.in_features
        net._fc = nn.Linear(num_ftrs, num_class)
    #net._dropout = nn.Dropout(0.7)
    net = net.to(device)
    earlt_stopping = EarlyStopping(save_path, patience, verbose=True)
    #net.load_state_dict(torch.load("./weights/Eb3_107_low_p.pth"))
    #net.load_state_dict(torch.load("./weights/" + net_class + "_" + str(num_epochs) + "/bestwork.pth"))
    #print(net)
    #设置损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)

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
        print("{}属正进行第{}轮训练".format(classname,epoch + 1))
        print("lr = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        #for i, data in tqdm(enumerate(train_dataloader)):
        with tqdm(train_dataloader,desc='EPOCH:'+str(epoch+1),colour='yellow') as datas:
            for data in datas:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                with amp.autocast(enabled=True):
                    outputs = net.forward(imgs)
                    loss = loss_func(outputs, targets)
                #outputs = net.forward(imgs)
                #loss = loss_func(outputs, targets)

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
        valid_acc = eval_acc / test_data_size
        earlt_stopping(valid_acc, net)
        if earlt_stopping.early_stop:
            print("Early Stopping")
            # ******************************** 模型文件保存位置 ******************************* 
            #torch.save(net.state_dict(), "./weights/Eb3_{}_low_p.pth".format(epoch + 1))
            print("Best acc model is saved!")
            break
        if((best_acc>=1)and((running_acc / train_data_size)>=1)):
            break
    end = time.time()
    print("Finish all epochs, best epoch: {}, best acc: {}".format(best_epoch, best_acc))
    print("running time:", end - start)
    epochs_list = [i for i in range(epoch+1)]
    plt.subplot(2, 2, 1)
    plt.title("loss line")
    plt.plot(epochs_list, loss_list)
    plt.subplot(2, 2, 2)
    plt.title("acc line")
    plt.plot(epochs_list, acc_list)
    plt.subplot(2, 2, 3)
    plt.title("test_loss line")
    plt.plot(epochs_list, t_loss_list)
    plt.subplot(2, 2, 4)
    plt.title("test_acc line")
    plt.plot(epochs_list, t_acc_list)
    plt.savefig(save_path+'/process_img.png')
    plt.show()
    writer.close()
