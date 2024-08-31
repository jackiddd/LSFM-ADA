import os  
from PIL import Image  
  
# 指定包含子目录的父文件夹路径  
parent_dir = 'expansion-dataset-10shot/image-to-text/'  # 替换为你的父文件夹路径  
  
# 遍历父文件夹下的所有子目录  
for subdir, dirs, files in os.walk(parent_dir):  
    for file in files:  
        # 检查文件是否是.png图片  
        if file.lower().endswith('.png'):  
            # 构建图片文件的完整路径  
            file_path = os.path.join(subdir, file)  
              
            # 打开图片文件  
            try:  
                img = Image.open(file_path)  
                  
                # 检查图片是否是灰度图，如果不是则转换为灰度图  
                if img.mode != 'L':  
                    img = img.convert('L')  
                  
                # 调整图片大小为64x64  
                img_resized = img.resize((64, 64))  
                  
                # 保存图片，覆盖原始文件  
                img_resized.save(file_path)  
                  
                print(f"Resized {file_path} to 64x64")  
            except Exception as e:  
                print(f"Error processing {file_path}: {e}")  
  
print("All images have been resized.")