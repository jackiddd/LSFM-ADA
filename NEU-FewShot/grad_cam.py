import torch  
import torch.nn as nn
import torchvision.models as models  
import torchvision.transforms as transforms  
from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder  
from torch.utils.data import DataLoader
from torch.autograd import Variable  
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, EigenCAM, XGradCAM, FullGrad  
from pytorch_grad_cam.utils.image import show_cam_on_image  
  
# 确保你的PyTorch模型是在评估模式下  
torch.set_grad_enabled(True)
  
# 数据预处理  
transform = transforms.Compose([    
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  

# 假设你有5个模型的路径  
model_paths = ['NEU-models/resnet50_5-shot.pth', 
               'NEU-models/resnet50_5-shot+cutout_expansion.pth', 
               'NEU-models/resnet50_5-shot+gridmask_expansion.pth', 
               'NEU-models/resnet50_5-shot+randaugment_expansion.pth',
               'NEU-models/resnet50_5-shot+expansion_image-to-text.pth'] 

# 加载测试数据集  
test_dataset = ImageFolder(root="NEU-data/gradcam-dataset/", transform=transform)  
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

for model_path, model_name in zip(model_paths, ['Baseline', 'Cutout', 'Gridmask', 'Randaugment', 'LSFM-ADA']):    
    resnet = models.resnet50(pretrained=False)  # 假设模型是自定义训练的，不使用预训练权重 
    resnet.fc =  nn.Linear(2048, 6)
    # 加载你的模型权重  
    resnet.load_state_dict(torch.load(model_path))  # 加载模型权重 
    resnet.fc = nn.Sequential()
    resnet = resnet.to(device)
    resnet.eval()    
    for index, (images, labels) in enumerate(test_dataloader):  
        # 使用GradCAM  
        images, labels = images.to(device), labels.to(device)
        grad_cam = GradCAM(model=resnet, target_layers=[resnet.layer4[-1]]) 
        # 获取特征图和梯度  
        grayscale_cam = np.squeeze(grad_cam(input_tensor=images, targets=None))
        image_tensor = images.squeeze(0)
        min_val = image_tensor.min()  
        max_val = image_tensor.max()
        epsilon = 1e-6  
        normalized_tensor = (image_tensor - min_val) / (max_val - min_val + epsilon) 
        images = np.transpose(normalized_tensor.cpu().numpy(), (1, 2, 0))
        # images = normalized_tensor
        # 将CAM图与原始图像叠加  
        visualization = show_cam_on_image(images, grayscale_cam, use_rgb=False, image_weight=0.6)
        save_dir = 'results/grad_cam/'+model_name+'/'+str(labels.item())+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imshow(visualization)
        plt.axis("off")
        plt.savefig(save_dir+'/'+str(index)+'.jpg', bbox_inches="tight", pad_inches=0.0)
  
