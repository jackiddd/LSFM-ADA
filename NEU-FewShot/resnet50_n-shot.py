import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader  
from PIL import Image  
import os


# 初始化数据集和数据加载器  
train_dataset = ImageFolder(root='/root/NEU-FewShot/data/10-shot-dataset/',
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))  # 替换[...]为你的数据预处理步骤

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  
# 加载预训练的Inception模型，并修改最后一层以匹配你的6个类别  
model = models.resnet50(pretrained=True)    
model.fc = nn.Linear(2048, 5)  # 修改为6个输出类别  
  
# 将模型移动到GPU（如果有的话）  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = model.to(device) 
  
# 定义损失函数和优化器  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
  
# 训练循环  
num_epochs = 20
for epoch in range(num_epochs):  
    running_loss = 0.0 
    running_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):  
        model.train()
        inputs, labels = inputs.to(device), labels.to(device)  
        # 梯度清零  
        optimizer.zero_grad()  
        # 前向传播  
        outputs = model(inputs)
        _, pred_result = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)  
        # 反向传播和优化  
        loss.backward()  
        optimizer.step()  
        # 累加损失  
        running_loss += loss.item() * inputs.size(0)
        running_acc += torch.sum(pred_result.data == labels.data).item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_acc / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, ACC: {epoch_acc:.4f}')

# 保存模型  
torch.save(model.state_dict(), '/root/NEU-FewShot/models/resnet50_10-shot.pth')