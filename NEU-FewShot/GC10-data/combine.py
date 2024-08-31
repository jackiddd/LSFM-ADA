import os  
import shutil       
  
  
def merge_dir(source_dirs, target_dir):
    if not os.path.exists(target_dir):  
        os.makedirs(target_dir)  

    for source_dir in source_dirs:  
        for subdir, dirs, files in os.walk(source_dir):  
            # 构造相对于源目录的路径  
            rel_path = os.path.relpath(subdir, source_dir)  
            
            # 在目标目录中创建相应的子目录  
            target_subdir = os.path.join(target_dir, rel_path)  
            if not os.path.exists(target_subdir):  
                os.makedirs(target_subdir)  
            
            # 遍历文件并复制到目标目录  
            for file in files:  
                source_file = os.path.join(subdir, file)  
                target_file = os.path.join(target_subdir, file)  
                
                # 如果目标文件不存在，则复制文件  
                if not os.path.exists(target_file):  
                    shutil.copy2(source_file, target_file)  
                else:  
                    # 如果文件已经存在，这里你可以选择跳过、覆盖或者合并（如果是文本文件的话）  
                    print(f"Warning: Skipping '{target_file}' because it already exists.") 
 
  
for item in ["label-to-text/", "thought-guide-chain/", "image-to-text/"]:
    merge_dir(["10-shot-dataset", "expansion-dataset-10shot/"+item], "10-shot+expansion-dataset/"+item)