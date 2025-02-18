import os
import shutil
lei = 'head'
class_num = 1
test_path = './'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_51_r_p/test'
train_path = './'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_51_r_p/train'
concat_test_data = os.listdir(test_path)#裁剪后的测试集数据
concat_train_data = os.listdir(train_path)#裁剪后的训练集数据
data_path = [test_path,train_path]
datalist=[concat_test_data,concat_train_data]
def mycopyfile(srcfile, dstpath):  # 移动函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 移动文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))
num=0
classnum =0
for i in range(2):
    if(i==0):
        temp = 'test'
    else:
        temp = 'train'
    for classname in datalist[i]:#进入第二级目录
        classnum+=1
        datalist2 = os.listdir(os.path.join(data_path[i],classname))
        for j in datalist2:
            new_name = j.split('#')[0]
            try:
                aug = new_name.split('_')[0]                    
                first_name = new_name.split('_')[1]
                second_name = j.split('#')[1]
            except:
                first_name = j.split('#')[0]
                second_name = j.split('#')[1]
            all_name = first_name#+' '+second_name属名
            mycopyfile(os.path.join(data_path[i],classname,j), os.path.join('./'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_18_r_p_noaug', temp, all_name + '/'))#将图片根据命名格式按种类分开
            num+=1
print("处理完成")
print(num)
print(classnum)
