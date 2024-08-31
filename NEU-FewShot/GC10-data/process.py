# import os  
# import shutil  
# import random  
  
# # 定义原始目录和新的训练/测试目录  
# base_dir = './'  # 原始目录路径  
# train_dir = './training-dataset'    # 训练集目录路径  
# test_dir = './testing-dataset'      # 测试集目录路径  
  
# # 如果新的目录不存在，则创建它们  
# if not os.path.exists(train_dir):  
#     os.makedirs(train_dir)  
# if not os.path.exists(test_dir):  
#     os.makedirs(test_dir)  
  
# # 遍历每个数据类目录  
# for class_dir in os.listdir(base_dir):  
#     class_path = os.path.join(base_dir, class_dir)  
#     if os.path.isdir(class_path):  
#         # 为当前数据类创建训练和测试子目录  
#         train_class_dir = os.path.join(train_dir, class_dir)  
#         test_class_dir = os.path.join(test_dir, class_dir)  
#         if not os.path.exists(train_class_dir):  
#             os.makedirs(train_class_dir)  
#         if not os.path.exists(test_class_dir):  
#             os.makedirs(test_class_dir)  
  
#         # 获取当前数据类下的所有jpg图片  
#         jpg_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]  
  
#         # 随机打乱图片顺序  
#         random.shuffle(jpg_files)  
  
#         # 按照3:1的比例分割为训练集和测试集  
#         split_index = int(len(jpg_files) * 0.75)  
#         train_files = jpg_files[:split_index]  
#         test_files = jpg_files[split_index:]  
  
#         # 复制文件到新的训练/测试目录，并按序号命名  
#         for i, file in enumerate(train_files, start=1):  
#             src_file = os.path.join(class_path, file)  
#             dst_file = os.path.join(train_class_dir, f'train_{i:04d}.jpg')  # 使用4位序号  
#             shutil.copy2(src_file, dst_file)  
  
#         for i, file in enumerate(test_files, start=1):  
#             src_file = os.path.join(class_path, file)  
#             dst_file = os.path.join(test_class_dir, f'test_{i:04d}.jpg')  # 使用4位序号  
#             shutil.copy2(src_file, dst_file)  
  
# print("训练和测试集文件已创建完成。")
import os  
import shutil  
import random  
  
# 假设 train_directory 是你之前生成的训练集目录  
train_dir = './training-dataset'  
sample_dir = './10-shot-dataset'  # 新的采样目录  
  
# 如果新的目录不存在，则创建它  
if not os.path.exists(sample_dir):  
    os.makedirs(sample_dir)  
  
# 遍历每个数据类目录  
for class_dir in os.listdir(train_dir):  
    class_path = os.path.join(train_dir, class_dir)  
    if os.path.isdir(class_path):  
        # 为当前数据类在采样目录中创建子目录  
        sample_class_dir = os.path.join(sample_dir, class_dir)  
        if not os.path.exists(sample_class_dir):  
            os.makedirs(sample_class_dir)  
  
        # 获取当前数据类下的所有jpg图片  
        jpg_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]  
  
        # 从jpg_files中随机选择5张图片  
        if len(jpg_files) >= 10:  
            sampled_files = random.sample(jpg_files, 10)  
        else:  
            sampled_files = jpg_files  # 如果没有足够的图片，则选择所有图片  
  
        # 复制文件到新的采样目录  
        for idx, file in enumerate(sampled_files):  
            src_file = os.path.join(class_path, file)  
            dst_file = os.path.join(sample_class_dir, "expanded_"+str(idx)+".jpg")  # 保持原文件名  
            shutil.copy2(src_file, dst_file)  
  
print("每类5张图片的采样已完成，并保存在", sample_dir)