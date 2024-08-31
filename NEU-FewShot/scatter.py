import torch  
import torch.nn as nn
import torchvision.models as models  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
from torchvision.datasets import ImageFolder  
import numpy as np  
from sklearn.manifold import TSNE  
import seaborn as sns  
import matplotlib.pyplot as plt  
  
# 假设你有一个ImageFolder数据集的路径  
test_data_dir = 'data/testing-dataset/'  
  
# 加载ResNet50模型并修改以获取倒数第二层的输出  
def get_feature_extractor(model_path):  
    resnet = models.resnet50(pretrained=False)  # 假设模型是自定义训练的，不使用预训练权重  
    # 加载你的模型权重  
    resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载模型权重  
    resnet.eval()  
      
    # 获取倒数第二层的输出  
    x = resnet.layer4[-1].output  
    x = x.view(x.size(0), -1)  # 展平特征  
    return torch.nn.Sequential(*list(resnet.children())[:-1], torch.nn.Flatten()) if x is None else torch.nn.Sequential(*list(resnet.children())[:-2], torch.nn.Sequential(resnet.layer4[-1], torch.nn.Flatten()))  
  
# 数据预处理  
transform = transforms.Compose([    
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  
  
# 加载测试数据集  
test_dataset = ImageFolder(root=test_data_dir, transform=transform)  
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  
  
# 假设你有5个模型的路径  
model_paths = ['models/resnet50_5-shot.pth', 
               'models/resnet50_5-shot+cutout_expansion.pth', 
               'models/resnet50_5-shot+gridmask_expansion.pth', 
               'models/resnet50_5-shot+randaugment_expansion.pth',
               'models/resnet50_expansion-finetuning_image-to-text.pth']  
  
# 提取每个模型的特征  
all_features = []  
all_labels = []
for model_path in model_paths:    
    label_list = []
    features = []  
    resnet = models.resnet50(pretrained=False)  # 假设模型是自定义训练的，不使用预训练权重 
    resnet.fc =  nn.Linear(2048, 6)
    # 加载你的模型权重  
    resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载模型权重 
    resnet.fc = nn.Sequential()
    resnet.eval()  
    with torch.no_grad():  
        for images, labels in test_dataloader:  
            feats = resnet(images).cpu().numpy()
            label_list.extend(labels.cpu().numpy())
            features.extend(feats)  
    all_features.append(np.array(features))  
    all_labels.append(np.array(label_list))
  
# 对每个模型的特征应用t-SNE降维  
tsne_results = []  
for feats in all_features:  
    tsne = TSNE(n_components=2, random_state=0)  
    tsne_results.append(tsne.fit_transform(feats))  
  
# 绘制四张散点图    
for i, (tsne_result, labels, model_name) in enumerate(zip(tsne_results, all_labels, ['Baseline', 'Cutout', 'Gridmask', 'Randaugment', 'LSFM-ADA']), 1):    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='deep')  
    plt.title(f't-SNE visualization of {model_name}', fontsize=20, fontweight='bold')    
    plt.tight_layout()  
    plt.savefig('results/scatter/'+model_name+'.jpg')
