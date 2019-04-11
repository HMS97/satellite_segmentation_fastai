from path import Path
from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
import os
from PIL import Image
import cv2
img_w = 256  
img_h = 256  

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
#     M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
#     xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
#     yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
        
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb
###crop part
def change_3_channel_to_1(mask, classes, prefix,index, ran):
    save_dir = './data'
    label_class = {}
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    (R,G,B) = cv2.split(mask)
    label_class['road'] = B
    label_class['water'] = G & R
    label_class['building'] = G - label_class['water']
    label_class['grass'] = R - label_class['water']
#     global split 

    if ran == True:
        string = '_random'
    else:
        string = ''
        

    for item in classes:
        mask = label_class[item]
        cv2.imwrite(f'{save_dir}/label/{item}/{prefix}{string}_{index}.png',mask)

    label_class['grass'][label_class['grass'] > 0] = 1
    label_class['road'][label_class['road'] > 0] = 2
    label_class['building'][label_class['building'] > 0] = 3
    label_class['water'][label_class['water'] > 0] = 4
    whole_map = label_class['water'] + label_class['building'] + label_class['road'] + label_class['grass'] 
    whole_map[whole_map > 4 ] = 0
    cv2.imwrite(f'{save_dir}/label/five_label/{prefix}{string}_{index}.png',whole_map)


def crop_by_sequence(image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir, classes ,same_scale = True):
    raw_img = cv2.imread(image_path,)
    raw_img_class = cv2.imread(img_class_path,)
    
    
    if same_scale == True:
        crop_size_w = crop_size_w * 2 
        crop_size_h = crop_size_h * 2
    
    print(raw_img.shape,raw_img_class.shape)


    h,w = raw_img.shape[0],raw_img.shape[1]

    index = 0
    x2,y2 = 0,0
    x0,y0 = 0,0
    while(y2<h):
        while(x2<w):
            x1 = x0
            x2 = x1 + crop_size_w
            y1 = y0
            y2 = y1 +crop_size_h


            if(x2>w or y2>h):
                x2 = min(x2,w)
                y2 = min(y2,h)
                if((x2-x1)>10 and (y2-y1)>10):
                    backgroud = np.zeros((crop_size_h,crop_size_w,raw_img.shape[2]),dtype=np.uint8)
                    backgroud[:y2-y1,:x2-x1] = raw_img[y1:y2,x1:x2]
                    patch = backgroud

                    backgroud_label = np.zeros((crop_size_h,crop_size_w,raw_img_class.shape[2]),dtype=np.uint8)
                    backgroud_label[:y2-y1,:x2-x1] = raw_img_class[y1:y2,x1:x2]
                    patch_label = backgroud_label
                else:
                    break
            else:
                patch = raw_img[y1:y2,x1:x2]
                patch_label = raw_img_class[y1:y2,x1:x2]
            #stride_h = auto_stride(patch_label)
            stride_h = crop_size_h
            stride_w = crop_size_w
            #print "current stride: ",stride_h
            x0 = x1 + stride_w
            
            if same_scale == True:
                patch = cv2.resize(patch,(int(crop_size_w/2), int(crop_size_h/2)))
                patch_label = cv2.resize(patch_label,(int(crop_size_w/2), int(crop_size_h/2)))

            
            success = cv2.imwrite(save_dir + f'/img/{prefix}_{index}.png',patch)
            change_3_channel_to_1(patch_label,classes, prefix,index,ran = 0)
            
            patch, patch_label = data_augment(patch,patch_label)

            
            prefix_aug = prefix + '_augment_'
            success_1 = cv2.imwrite(save_dir + f'/img/{prefix_aug}_{index}.png',patch)
            change_3_channel_to_1(patch_label,classes, prefix_aug,index,ran = 0 )
            if success == 1 and success_1 ==1 :
                pass
            else:
                print('seq_save err')
            index = index + 1
        x0,x1,x2 = 0,0,0
        y0 = y1 + stride_h

        
        
def crop_by_random(num,image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir, classes, same_scale = True ):
    
    if same_scale == True:
        crop_size_w = crop_size_w * 2 
        crop_size_h = crop_size_h * 2

    raw_img = cv2.imread(image_path,)
    raw_img_class = cv2.imread(img_class_path)
    print(raw_img.shape, raw_img_class.shape)
    h,w = raw_img.shape[0],raw_img.shape[1]
    index = 0 
    range_h = h - crop_size_h - 1
    range_w = w - crop_size_w - 1
    
    list_x = np.random.randint(low = 0, high = range_h, size = num)
    list_y = np.random.randint(low = 0, high = range_w, size = num)
    combine = list(zip(list_x,list_y))
    for i in combine:
        
        patch = raw_img[i[0]:i[0] + crop_size_h, i[1]:i[1] + crop_size_w]
        patch_label = raw_img_class[i[0]:i[0] + crop_size_h, i[1]:i[1] + crop_size_w]
        
        if same_scale == True:
            patch = cv2.resize(patch,(int(crop_size_w/2), int(crop_size_h/2)))
            patch_label = cv2.resize(patch_label,(int(crop_size_w/2), int(crop_size_h/2)))

        success = cv2.imwrite(save_dir + f'/img/{prefix}_random_{index}.png',patch)
        change_3_channel_to_1(patch_label,classes, prefix,index,ran = 1 )
        
        patch, patch_label = data_augment(patch,patch_label)

        prefix_aug = prefix + '_augment_'
        
        success_1 = cv2.imwrite(save_dir + f'/img/{prefix_aug}_random_{index}.png',patch)
        change_3_channel_to_1(patch_label,classes, prefix_aug,index,ran = 1)
        if success == 1 and success_1 ==1 :
                pass
        else:
            print('random save err', success, success_1)

        index = index + 1

        
        
        


def generate(num = 10000,split = False, crop_size_h = 512, crop_size_w = 512, save_dir = './data',string = '', same_scale = True):
    
    print(crop_size_h, crop_size_w)
    
        
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(f'{save_dir}/img'):
        os.mkdir(f'{save_dir}/img')
    if not os.path.exists(f'{save_dir}/label'):
        os.mkdir(f'{save_dir}/label')
        os.mkdir(f'{save_dir}/label/road')
        os.mkdir(f'{save_dir}/label/water')
        os.mkdir(f'{save_dir}/label/grass')
        os.mkdir(f'{save_dir}/label/building')
        os.mkdir(f'{save_dir}/label/five_label')

    classes = ['road','water', 'grass','building']
    
    for i in range(1,6):

        image_path = Path('./BDCI2017-seg/CCF-training-Semi')/f'{i}.png'
        img_class_path = Path('./BDCI2017-seg/CCF-training-Semi')/ f'{i}_class_vis.png'
        prefix = f"picture_{i}"
        prefix = string + prefix
        print(image_path)
        print(img_class_path)
        crop_by_random(num,image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir, classes,same_scale = same_scale )
        crop_by_sequence(image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir, classes, same_scale = same_scale)

    classes = ['road','water', 'grass','building']

    for i in classes:
            print(len(Path(f'./data/label/{i}').files()))
    print(len(Path('data/img').files()))
    
# if __name__ == "__main__":
#     main()
