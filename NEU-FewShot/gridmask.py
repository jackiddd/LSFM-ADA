import torch  
import torchvision.transforms as transforms  
from PIL import Image  
import numpy as np  
import os 
  

import numpy as np  
from PIL import Image  
  
class GridMask(object):  
    def __init__(self, size, ratio=0.5, fill_value=0):  
        self.size = size  
        self.ratio = ratio  
        self.fill_value = fill_value  
  
    def __call__(self, img):  
        h, w = img.size  
        Hh, Ww = int(h * self.ratio), int(w * self.ratio)  
        sx, sy = np.random.randint(0, h - Hh), np.random.randint(0, w - Ww)  
  
        # Create grid mask  
        grid_mask = np.zeros((h, w), dtype=np.float32)  
        for i in range(0, Hh, 16):  
            for j in range(0, Ww, 16):  
                # 这里可以定义不同的模式，比如只遮挡奇数或偶数位置的网格  
                grid_mask[sx + i:sx + i + 8, sx + j:sx + j + 8] = 1  
  
        # Apply grid mask to image  
        img_np = np.array(img)  
        img_np[grid_mask == 1] = self.fill_value  
  
        return Image.fromarray(np.uint8(img_np))  
  

data_dir = '/root/NEU-FewShot/data/5-shot-dataset/'  
output_dir = '/root/NEU-FewShot/data/gridmask_expansion-dataset/'  
  
# 确保输出目录存在  
if not os.path.exists(output_dir):  
    os.makedirs(output_dir)  
  
# 遍历每个类别目录  
for class_dir in os.listdir(data_dir):  
    class_path = os.path.join(data_dir, class_dir)  
    if os.path.isdir(class_path):  
        # 创建该类的输出目录  
        output_class_dir = os.path.join(output_dir, class_dir)  
        if not os.path.exists(output_class_dir):  
            os.makedirs(output_class_dir)  
          
        # 遍历该类中的每张图像  
        for img_file in os.listdir(class_path):  
            img_path = os.path.join(class_path, img_file)  
            if img_file.endswith('.png') or img_file.endswith('.jpg'):  # 假设图像是.png或.jpg格式  
                # 读取图像  
                img = Image.open(img_path).convert('L')  # 转换为灰度图  
                  
                # 应用Cutout变换并保存新图像  
                for i in range(20):      
                    gridmask_transform = GridMask(size=64, ratio=0.9, fill_value=0)  
                    augmented_img = gridmask_transform(img)    
                      
                    # 生成唯一的文件名并保存图像  
                    output_filename = f"{os.path.splitext(img_file)[0]}_gridmask_{i}.png"  
                    output_img_path = os.path.join(output_class_dir, output_filename)  
                    augmented_img.save(output_img_path)  

                img.save(os.path.join(output_class_dir, img_file))
  