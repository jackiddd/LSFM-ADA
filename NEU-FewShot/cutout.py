import torch  
import torchvision.transforms as transforms  
from PIL import Image  
import numpy as np  
import os 
  

class Cutout(object):  
    def __init__(self, length):  
        self.length = length  
  
    def __call__(self, img):  
        W, H = img.size  
        t = np.random.rand(1) * (W * H - self.length ** 2)  
        x = int(np.sqrt(t) % W)  
        y = int(t // W)  
  
        xy1 = np.clip(np.array([x, y, x + self.length, y + self.length]), 0, [W, H, W, H])  
        xy2 = [max(0, xy1[0]), max(0, xy1[1]), min(W, xy1[2]), min(H, xy1[3])]  
  
        img = np.array(img)
        img[xy2[0]:xy2[2], xy2[1]:xy2[3]] = 0  
        return Image.fromarray(np.uint8(img))  
    
  
data_dir = '/root/NEU-FewShot/data/5-shot-dataset/'  
output_dir = '/root/NEU-FewShot/data/cutout_expansion-dataset/'  
  
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
                    # 假设你想要一个16x16的Cutout区域  
                    cutout_transform = Cutout(48)
                    augmented_img = cutout_transform(img)    
                      
                    # 生成唯一的文件名并保存图像  
                    output_filename = f"{os.path.splitext(img_file)[0]}_cutout_{i}.png"  
                    output_img_path = os.path.join(output_class_dir, output_filename)  
                    augmented_img.save(output_img_path)  

                # img.save(os.path.join(output_class_dir, img_file))
  