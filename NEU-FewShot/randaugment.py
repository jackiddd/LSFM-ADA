import random  
import os
from PIL import Image, ImageEnhance  
from torchvision import transforms  
  
def get_grayscale_transforms():  
    # 定义灰度图像的增强策略  
    transforms_list = [  
        transforms.RandomHorizontalFlip(p=0.6),  # 水平翻转  
        transforms.RandomVerticalFlip(p=0.6),   # 垂直翻转  
        transforms.RandomCrop((32,32)),  # 随机裁剪
        transforms.Resize((64,64)),
        transforms.RandomRotation(degrees=10),   # 随机旋转  
          
        # 亮度调整  
        lambda img: ImageEnhance.Brightness(img).enhance(  
            1 + random.uniform(-0.5, 0.5)  # 亮度因子在0.6到1.4之间  
        ),  
          
        # 对比度调整  
        lambda img: ImageEnhance.Contrast(img).enhance(  
            1 + random.uniform(-0.5, 0.5)  # 对比度因子在0.6到1.4之间  
        ),   
    ]  
  
  
    return transforms.Compose(transforms_list)  
  

data_dir = '/root/NEU-FewShot/data/5-shot-dataset/'  
output_dir = '/root/NEU-FewShot/data/randaugment_expansion-dataset/'  
  
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
                    grayscale_transforms = get_grayscale_transforms()      
                    augmented_img = grayscale_transforms(img)    
                    # 生成唯一的文件名并保存图像  
                    output_filename = f"{os.path.splitext(img_file)[0]}_randaugment_{i}.png"  
                    output_img_path = os.path.join(output_class_dir, output_filename)  
                    augmented_img.save(output_img_path)  

                # img.save(os.path.join(output_class_dir, img_file))