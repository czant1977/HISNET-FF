import os
import shutil

import cv2
import numpy as np
from PIL import Image
import random as r
r.seed(7)
def cutout(ipath, n_holes, length,epoch=0):
    """
    Randomly mask out one or more patches from an imag
    :param ipath: img_path
    :param n_holes:  Number of patches to cut out of each image
    :param length: The length (in pixels) of each square patch
    :return: save_img_path
    """
    save_path = './ic_temp/cutout'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    # save path
    img_save_name = 'cutout'+str(epoch)+'_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    h, w = img.shape[:2]
    for n in range(n_holes):
        # 正方形区域中心点随机出现
        y = np.random.randint(h)
        x = np.random.randint(w)
        # 划出正方形区域，边界处截断
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        # 全0填充
        img[y1: y2, x1: x2] = 0
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: cutout\t img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path


def rotation(ipath, angle):
    """
    :param ipath: img_path
    :param angle: angle
    :return: img_save_path
    """
    save_path = './ic_temp/rot' + '_' + str(angle)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    # save path
    img_save_name = 'rot' + str(angle) + '_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # # 生成旋转矩阵，参数依次为旋转中心（center）、角度（angle）、缩放比例（scale）
    # h, w = img.shape[:2]
    # M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # # 进行旋转变换
    # img = cv2.warpAffine(img, M, (h, w), borderValue=(255, 0, 0))
    # save
    # if(angle==270):
    #     img = cv2.flip(img, 1)
    #     img = cv2.transpose(img)
    # elif(angle==90):
    #     img = cv2.transpose(img)
    #     img = cv2.flip(img, 1)
    # else:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 进行旋转变换
    img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
    cv2.imwrite(img_save_path, img)
    print("Operation: rot{}\t img: {}".format(str(angle), os.path.split(ipath)[-1]))
    return img_save_path

def txy(ipath, tx, ty):
    """
    :param ipath: img_path
    :param tx: distance x for translation
    :param ty: distance y for tanslation
    :return: img_save_path
    """
    save_path = './ic_temp/tx' + str(tx) + 'ty' + str(ty)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    # save path
    img_save_name = 'tx' + str(tx) + 'ty' + str(ty) + '_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # translation
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
    # save
    cv2.imwrite(img_save_path, img)
    print("Operation: tx{}ty{}\t img: {}".format(str(tx), str(ty), os.path.split(ipath)[-1]))
    return img_save_path


def guassNoise(ipath, mean=0, st=1):
    """
    :param ipath: img_path
    :param mean: default mean=0
    :param st: default standard deviation=1
    :return: img_save_path
    """
    save_path = './ic_temp/gn'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    # save path
    img_save_name = 'gn' + '_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # creat noise
    noise = np.random.normal(mean, st, img.shape)
    #	一定要将噪声的类型转换为uint
    noise = noise.astype('uint8')
    img = cv2.add(img, noise)
    # save
    cv2.imwrite(img_save_path, img)
    print("Operation: guassNoise\t img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path
def salt_and_pepper_noise(image_path, salt_prob=0.02, pepper_prob=0.02):
    """
    :param image_path: Input image path.
    :param salt_prob: Probability of adding salt noise.
    :param pepper_prob: Probability of adding pepper noise.
    :return: Image with salt and pepper noise.
    """
    save_path = './ic_temp/spn'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img = cv2.imread(image_path)
    
    # save path
    img_save_name = 'spn' + '_' + os.path.split(image_path)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    
    # Generate random matrix with the same shape as the image
    salt_matrix = np.random.rand(*img.shape[:2])
    pepper_matrix = np.random.rand(*img.shape[:2])
    
    # Add salt noise
    img[salt_matrix < salt_prob] = [255, 255, 255]  # White pixels
    
    # Add pepper noise
    img[pepper_matrix < pepper_prob] = [0, 0, 0]  # Black pixels
    
    # Save the image
    cv2.imwrite(img_save_path, img)
    
    print("Operation: salt_and_pepper_noise\t img: {}".format(os.path.split(image_path)[-1]))
    
    return img_save_path
def poisson_noise(image_path, scale_factor=1.0):
    """
    :param image_path: Input image path.
    :param scale_factor: Scaling factor for adjusting noise intensity.
    :return: Image with Poisson noise.
    """
    save_path = './ic_temp/pn'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img = cv2.imread(image_path)

    # save path
    img_save_name = 'pn' + '_' + os.path.split(image_path)[-1]
    img_save_path = os.path.join(save_path, img_save_name)

    # Generate Poisson noise
    noise = np.random.poisson(img * scale_factor) / scale_factor
    img_with_noise = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Save the image
    cv2.imwrite(img_save_path, img_with_noise)

    print("Operation: poisson_noise\t img: {}".format(os.path.split(image_path)[-1]))

    return img_save_path
# def elastic_deformation(image_path, alpha=200, sigma=20):
#     """
#     :param image_path: Input image path.
#     :param alpha: Scaling factor for controlling the intensity of deformation.
#     :param sigma: Standard deviation for Gaussian filter.
#     :return: Image with elastic deformation.
#     """
#     save_path = './ic_temp/ed'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     img = cv2.imread(image_path)

#     # save path
#     img_save_name = 'ed' + '_' + os.path.split(image_path)[-1]
#     img_save_path = os.path.join(save_path, img_save_name)

#     # Generate elastic deformation
#     random_state = np.random.RandomState(None)
#     shape = img.shape
#     dx = alpha * np.random.normal(size=(shape[0], shape[1]), scale=sigma)
#     dy = alpha * np.random.normal(size=(shape[0], shape[1]), scale=sigma)

#     x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#     indices = np.reshape(y+dy, (-1, 1)).astype('float32'), np.reshape(x+dx, (-1, 1)).astype('float32')

#     distorted_img = cv2.remap(img, indices[1], indices[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

#     # Save the image
#     cv2.imwrite(img_save_path, distorted_img)

#     print("Operation: elastic_deformation\t img: {}".format(os.path.split(image_path)[-1]))

#     return img_save_path
def random_brightness(ipath, factor=0.5):
    # 随机调整亮度
    save_path = './ic_temp/brightness'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
     # 将图像从BGR格式转换为HSV格式
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 随机生成亮度调整因子，可以根据需要调整范围
    brightness_factor = r.uniform(1 - factor, 1 + factor)

    # 调整亮度
    hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255)

    # 将图像从HSV格式转换回BGR格式
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_save_name = 'brightness_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: brightness img: {}".format(os.path.split(ipath)[-1]))

    return img_save_path
def random_contrast(ipath,factor=0.5):
    save_path = './ic_temp/contrast'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    contrast_factor = r.uniform(1 - factor, 1 + factor)

    # 调整对比度
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
    img_save_name = 'contrast_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: contrast img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path
def random_hue(ipath,factor=0.5):
    save_path = './ic_temp/hue'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 随机生成色调调整因子，可以根据需要调整范围
    hue_factor = r.uniform(-factor, factor)

    # 调整色调
    hsv[:,:,0] = (hsv[:,:,0] + hue_factor) % 180

    # 将图像从HSV格式转换回BGR格式
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_save_name = 'hue_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: hue img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path
def random_saturation(ipath):
    save_path = './ic_temp/saturation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 随机生成饱和度的缩放因子
    saturation_factor = r.uniform(0.5, 1.5)

    # 将饱和度缩放应用于HSV图像的第二个通道（即饱和度通道）
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

    # 将图像从HSV转换回BGR颜色空间
    img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    img_save_name = 'saturation_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: saturation img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path
def Local_rotation(ipath):
    save_path = './ic_temp/local_rotation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)
    #image = cv2.imread('your_image_path.jpg')

# 确保图像大小是480x480
    img = cv2.resize(img, (480, 480))

    # 将图像分成4块
    height, width = img.shape[:2]
    half_height, half_width = height // 2, width // 2

    # 分割图像成四块
    top_left = img[:half_height, :half_width]
    top_right = img[:half_height, half_width:]
    bottom_left = img[half_height:, :half_width]
    bottom_right = img[half_height:, half_width:]

    # 对每块进行翻转
    top_left = cv2.flip(top_left, 1)  # 水平翻转
    top_right = cv2.flip(top_right, 0)  # 垂直翻转
    bottom_left = cv2.flip(bottom_left, -1)  # 水平和垂直翻转
    bottom_right = cv2.flip(bottom_right, 1)  # 水平翻转

    # 拼接四块图像
    img = np.concatenate((np.concatenate((top_left, top_right), axis=1),
                            np.concatenate((bottom_left, bottom_right), axis=1)), axis=0)
    img_save_name = 'lr_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: local_rotation img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path
def self_augumentation(ipath,size=200):
    crop_size=(size, size)
    save_path = './ic_temp/self_augumentation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(ipath)

    # 获取原始图像的高度和宽度
    height, width = img.shape[:2]

    # 随机生成裁剪的起始位置
    top = r.randint(0, height - crop_size[0])
    left = r.randint(0, width - crop_size[1])

    # 随机裁剪图像
    img2 = img[top:top + crop_size[0], left:left + crop_size[1]]

    # 随机生成粘贴的位置
    paste_top = r.randint(0, height - crop_size[0])
    paste_left = r.randint(0, width - crop_size[1])

    # 创建一个与原始图像相同大小的图像，将裁剪的图像粘贴到随机位置
    #img = img.copy()
    img[paste_top:paste_top + crop_size[0], paste_left:paste_left + crop_size[1]] = img2
    img_save_name = 'crop'+str(size)+'_'+ os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: self_aug img: {}".format(os.path.split(ipath)[-1]))
    return img_save_path
def flip(ipath, direction):
    """
    :param ipath: img_path
    :param direction: (str)['up-down', 'lef-right', mixture]
    :return: img_save_path
    """
    save_path = "./ic_temp/flip_" + direction
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    flip_dict = {'up-down': 0, 'left-right': 1, 'mixture': -1}
    img = cv2.imread(ipath)
    # flip
    img = cv2.flip(img, flip_dict[direction])
    # save path
    img_save_name = 'flip' + str(flip_dict[direction] + 1) + '_' + os.path.split(ipath)[-1]
    img_save_path = os.path.join(save_path, img_save_name)
    # save img
    cv2.imwrite(img_save_path, img)
    print("Operation: flip_direction: {}\t img: {}".format(direction, os.path.split(ipath)[-1]))
    return img_save_path




if __name__ == '__main__':
    c_path = os.getcwd()
    #print("工作环境目录：{}".format(c_path))
    '''
    if not os.path.exists(os.path.join(c_path, 'bysj-od-train_aug')):
        shutil.copytree(os.path.join(c_path, 'bysj-od-train'),
                        os.path.join(c_path, 'bysj-od-train_aug'))
    '''
    kind_path = os.path.join('./head_data/head_2level_r_p')
    kind_file = os.listdir(kind_path)
    for d in kind_file:  # 一级目录Condylura等
        train_path = os.path.join(kind_path, d, 'train')
        train_dir = os.listdir(train_path)
        for k in train_dir:#train目录下的种类
            img_path = os.path.join(train_path,k)
            imgs = os.listdir(img_path)
            for i in imgs:  # i: Euroscaptor#grandis#57110#grandis_57110_DSC_3659#s#v.JPG
                ipath = os.path.join(img_path, i)
                copy_to_path = img_path + '/'
                img = cutout(ipath, n_holes=10, length=40,epoch=1)
                shutil.copy(img, copy_to_path)
                # img = cutout(ipath, n_holes=10, length=40,epoch=2)
                # shutil.copy(img, copy_to_path)
                # img = cutout(ipath, n_holes=10, length=40,epoch=3)
                # shutil.copy(img, copy_to_path)
                # img = rotation(ipath, 270)
                # shutil.copy(img, copy_to_path)
                img = rotation(ipath, 90)
                shutil.copy(img, copy_to_path)
                img = rotation(ipath, 180)
                # shutil.copy(img, copy_to_path)
                # img = rotation(ipath, 45)
                shutil.copy(img, copy_to_path)
                img = txy(ipath, 40, 0)
                # shutil.copy(img, copy_to_path)
                # img = txy(ipath, 0, 200)
                shutil.copy(img, copy_to_path)
                img = guassNoise(ipath, 0, 1)
                shutil.copy(img, copy_to_path)
                img = random_brightness(ipath)
                shutil.copy(img, copy_to_path)
                # img = random_saturation(ipath)
                # shutil.copy(img, copy_to_path)
                # img = random_contrast(ipath)
                # shutil.copy(img, copy_to_path)
                img = random_hue(ipath)
                shutil.copy(img, copy_to_path)
                # img = salt_and_pepper_noise(ipath)
                # shutil.copy(img, copy_to_path)
                # img = poisson_noise(ipath)
                # shutil.copy(img, copy_to_path)
                img = Local_rotation(ipath)
                shutil.copy(img, copy_to_path)
                img = self_augumentation(ipath,size=200)
                shutil.copy(img, copy_to_path)
                # img = self_augumentation(ipath,size=300)
                # shutil.copy(img, copy_to_path)
                # img = elastic_deformation(ipath)
                # shutil.copy(img, copy_to_path)
                # img = zoom(ipath, 0.8)
                # shutil.copy(img, copy_to_path)
                # img = flip(ipath, 'mixture')
                # shutil.copy(img, copy_to_path)
                # img = flip(ipath, 'left-right')
                # shutil.copy(img, copy_to_path)
                # img = flip(ipath, 'up-down')
                # shutil.copy(img, copy_to_path)

                # # show
                # img = cv2.imread(img)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = Image.fromarray(img)
                # img.show()
                # break
