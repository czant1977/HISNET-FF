import os
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import time
from wepub import SendMessage
from sklearn.preprocessing import MinMaxScaler
from MLP_model import MLP
sm = SendMessage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = []
        self.labels = []
        
        # 读取数据文件
        with open(data_file, 'r') as f:
            line_num=0
            for line in f:
                #print(line)
                line_num+=1
                #print(line_num)
                line = eval(line.strip())#.split('\n')  # 假设数据以空格分隔
                sequence = [float(x) for x in line]
                #print(len(sequence))
                self.data.append(sequence)
        
        # 读取标签文件
        with open(label_file, 'r') as f:
            for line in f:
                label = int(line.strip())
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(data), torch.tensor(label)
    def get_class_to_idx(self):
        # 返回类别到标签的映射关系
        return self.class_to_idx
batch_size = 128
num_class= 51
shu_list =['51']#['Scapanus','Mogera','Scaptonyx','Uropsilus','Euroscaptor','Parascaptor','Talpa']#,'49']
lei='head'          #teeth,head,concat_5:1,concat
class_num = '1'
server_name = 'b7_'+class_num


net = MLP(num_class=num_class).to(device)
for classnum,shu in enumerate(shu_list):
    start_time = time.time()
    file_path = './'+server_name+'/'+shu+'/'+lei+'_data/'
    x_train_path = file_path+'x_train.txt'
    y_train_path = file_path+'y_train.txt'
    x_test_path = file_path+'x_test.txt'
    y_test_path = file_path+'y_test.txt'
    train_dataset = CustomDataset(x_train_path, y_train_path)
    test_dataset = CustomDataset(x_test_path, y_test_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print(train_dataset.get_class_to_idx)
    print('训练集数据量为:{}\n测试集数量为:{}'.format(train_data_size, test_data_size))
    name_style = shu+'/'+lei
    if os.path.exists("./weights-"+server_name+"/"+name_style) is False:
        os.makedirs("./weights-"+server_name+"/"+name_style)
    try:
        net = torch.load('./weights-'+server_name+"/18/"+lei+'/best_model2.pkl',map_location=device)
    except:
        pass
    save_path = "./weights-"+server_name+"/"+name_style
    num_epochs = 100
    step_size = 20
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)
    learning_rate = 0.01
    #设置优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.001)#, momentum=0.9)
    #设置学习率迭代器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8, last_epoch=-1)
    best_acc=0
    for epoch in range(num_epochs):
        net.train()
        net.to(device)
        total_train_loss = 0
        running_acc = 0
        print("——————————————————————————————{}——————————————————————————————".format(epoch + 1))
        print("lr = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        for i,(data,labels) in tqdm(enumerate(train_dataloader)):
            imgs, targets = data,labels
            #print(imgs[1])
            #imgs = torch.tensor(imgs).view(1,4096)
            imgs = imgs.unsqueeze(1).unsqueeze(2).to(device)
            targets = targets.to(device)
            outputs = net(imgs)
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
        scheduler.step()
        train_acc = running_acc / train_data_size
        print('{}属Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(shu,
            epoch + 1, total_train_loss / train_data_size, train_acc))
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.unsqueeze(1).unsqueeze(2).to(device)
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
                best_epoch=epoch+1
                f = open("./weights-"+server_name+"/"+name_style+'/'+'best_acc.txt',mode = 'w')
                f.write(str(train_dataset.get_class_to_idx)+'\n')
                f.write(('训练集数据量为:{}\n，测试集数量为:{}\n'.format(train_data_size, test_data_size)))
                f.write('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}\n'.format(
            epoch + 1, total_train_loss / train_data_size, running_acc / train_data_size))
                f.write('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_size
                                                        , eval_acc / test_data_size))
                f.close()
                torch.save(net,'./weights-'+server_name+"/"+name_style+'/'+'best_model.pkl')
                if((best_acc==1)and (train_acc)==1):
                    break
        if(epoch+1-best_epoch>=50):
            print('50轮未增长，停止训练')
            break
    print('best_epoch:{},best_acc:{:.6f},final_acc:{:.6f}'.format(best_epoch,best_acc,eval_acc/test_data_size))
    torch.save(net,'./weights-'+server_name+"/"+name_style+'/'+'final_model.pkl')
    end_time = time.time()
    all_time = end_time - start_time
    sm = SendMessage()
    json_data = {
        "shu": shu,
        "train_acc": train_acc,
        "test_acc": best_acc,
        "epoch": best_epoch,
        "second": all_time,
    }
    # 发送消息
    sm.send_message(json_data=json_data)