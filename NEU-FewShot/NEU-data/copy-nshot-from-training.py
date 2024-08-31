import os  
import shutil  
import random  
  
# 源目录和目标目录  
source_dir = 'training-dataset/'  
target_dir = '10-shot-dataset/'  
  
# 确保目标目录存在  
if not os.path.exists(target_dir):  
    os.makedirs(target_dir)  
  
# 遍历源目录中的所有子目录  
for subdir in os.listdir(source_dir):  
    if os.path.isdir(os.path.join(source_dir, subdir)):  
        # 构造源子目录和目标子目录的路径  
        source_subdir_path = os.path.join(source_dir, subdir)  
        target_subdir_path = os.path.join(target_dir, subdir)  
          
        # 如果目标子目录不存在，则创建它  
        if not os.path.exists(target_subdir_path):  
            os.makedirs(target_subdir_path)  
          
        # 获取源子目录下所有jpg图片的文件名  
        image_files = [f for f in os.listdir(source_subdir_path) if f.endswith('.jpg')]  
          
        # 如果没有图片，则跳过  
        if not image_files:  
            continue  
          
        # 均匀地选择n个文件名  
        selected_files = image_files[0:200:20]  
          
        # 将选中的图片复制到目标子目录中  
        for idx, file in enumerate(selected_files):  
            source_file_path = os.path.join(source_subdir_path, file)  
            target_file_path = os.path.join(target_subdir_path, "expanded_"+str(idx)+".jpg")  
            shutil.copy2(source_file_path, target_file_path)  # 使用copy2保留元数据  
  
print("图片复制完成。")