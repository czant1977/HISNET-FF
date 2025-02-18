import os
import cv2
import numpy as np


# 坐标转换，原始存储的是YOLOv5格式[x_center, y_center, width, height]
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2]
# where xy1=top--left, xy2=bottom--right
px=600
def xywhn2xyxy(x, w, h, padw=0, padh=0):
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def resize_with_padding(img, size):
    # hw_scale = 0.0000   # 高/宽的比例系数
    h, w = img.shape[0:2]  # (height, width, channel)
    hw_scale = h / w
    if h >= w:
        h_n = size
        w_n = h_n / hw_scale
        top = bottom = 0
        left = right = int((h_n - w_n) / 2)
        if not (h_n - int(w_n) - left * 2 == 0):
            # left = right = right + int((h_n - int(w_n) - left * 2) / 2)
            right += h_n - int(w_n) - left * 2
    else:
        w_n = size
        h_n = w_n * hw_scale
        top = bottom = int((w_n - h_n) / 2)
        left = right = 0
        if not (w_n - int(h_n) - top * 2 == 0):
            # top = bottom = bottom + int((w_n - int(h_n) - bottom * 2) / 2)
            bottom += w_n - int(h_n) - top * 2
    # resize
    img = cv2.resize(img, (int(w_n), int(h_n)))  # (width, height)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img


def concat_base_box(ipath, lpath,to_dir):
    save_path=to_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # read label.txt
    with open(lpath, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
    img = cv2.imread(ipath)
    img_name = os.path.split(ipath)[-1]
    h, w = img.shape[:2]
    # create background image
    bgi = np.zeros_like(img, np.uint8)
    bgi[:] = [255, 0, 0]  # BGR 纯色图
    #cv2.imwrite(save_path+'/1.JPG', bgi)
    # obtain original img coordinate
    lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化
    lb_origin = lb[:]
    lb_bgi = lb[:]
    for i, x in enumerate(lb_bgi):
        # 根据原图坐标裁剪图片
        xx1, yy1, xx2, yy2 = lb_origin[i, 1:]
        #print(lb_origin[i,0])
        crop_img = img[int(yy1):int(yy2), int(xx1):int(xx2)]
        # 缩放后拼接
        x1, y1, x2, y2 = x[1:]
        crop_img = cv2.resize(crop_img, (int(x2) - int(x1), int(y2) - int(y1)))
        # if(i%2==0):
        #     bgi[int(y1):int(y2), int(x1):int(x2)] = crop_img
        # else:

        bgi[int(y1):int(y2), int(x1):int(x2)] = crop_img
        bgi = bgi.astype(np.uint8)
    cv2.imwrite(save_path + '/ot'+img_name+'.JPG', bgi)
    # 仅裁剪含有牙齿的区域
    x_min = np.amin(lb_bgi[:, 1])
    y_min = np.amin(lb_bgi[:, 2])
    x_max = np.amax(lb_bgi[:, 3])
    y_max = np.amax(lb_bgi[:, 4])
    bgi = bgi[int(y_min):int(y_max), int(x_min):int(x_max)]
    origin_shape = bgi.shape
    bgi = cv2.copyMakeBorder(bgi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    # resize with padding
    #bgi = resize_with_padding(bgi, px)
    # save
    save_path = os.path.join(save_path, 'teeth_'+img_name)
    cv2.imwrite(save_path, bgi)
    # print
    print("img: {}\t origin_shape: {}".format(img_name, origin_shape))
    return save_path

def concat(img_path,label_path,to_dir):
    teeth_path = concat_base_box(img_path,label_path,to_dir)
    return teeth_path
# concat(r'E:\final_mouse\Crocudura_data\Crocudura_data\images\Euroscaptor#klossi#AMNH87314#klossi_AMNH87314_DSC_5359#s#v.JPG',
#         r'E:\final_mouse\Crocudura_data\Crocudura_data\labels\Euroscaptor#klossi#AMNH87314#klossi_AMNH87314_DSC_5359#s#v.txt',
#         r'E:\final_mouse\Crocudura_data\Crocudura_data')
head_path = 'D:/yolo-touk/photo/other_wuzhong/head'
label_path = 'D:/yolo-touk/photo/other_wuzhong/label'
t_path = 'D:/yolo-touk/photo/other_wuzhong/teeth'
for img in os.listdir(head_path):
    img_path = os.path.join(head_path,img)
    label = os.path.join(label_path,img.split('.')[0]+'.txt')
    concat(img_path,label,t_path)