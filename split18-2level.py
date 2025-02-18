import os
import shutil
lei = 'teeth'
class_num = 3
path = './'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_18_r_p'
concat_test_data = os.listdir(path + '/test')#裁剪后的测试集数据
concat_train_data = os.listdir(path + '/train')#裁剪后的训练集数据
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
for i,temp in enumerate(['test','train']):
#temp='train'
    for k in datalist[i]:#18属
        class_name_lsit = os.listdir(os.path.join(path,temp,k))

        for j in class_name_lsit:#每个属文件夹中的图片
            new_name = j.split('#')[0]
            try:
                aug = new_name.split('_')[0]                    
                first_name = new_name.split('_')[1]
                second_name = j.split('#')[1]
            except:
                first_name = j.split('#')[0]
                second_name = j.split('#')[1]
            all_name = first_name+' '+second_name
            # if not os.path.exists(os.path.join('./'+lei+'_5zhe_data/head_data_'+str(class_num)+'/'+lei+'_2level_r_p', k, 'test', all_name)):
            #     os.makedirs(os.path.join('./'+lei+'_5zhe_data/head_data_'+str(class_num)+'/'+lei+'_2level_r_p', k, 'test', all_name))
            if(i==1):
                if not os.path.exists(os.path.join('./'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_2level_r_p', k, 'test', all_name)):
                    continue
            mycopyfile(os.path.join(path,temp,k,j), os.path.join('./'+lei+'_5zhe_data/'+lei+'_data_'+str(class_num)+'/'+lei+'_2level_r_p', k, temp, all_name + '/'))#将图片根据命名格式按种类分开
        print("处理完成")

