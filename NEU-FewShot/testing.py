import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader


# 加载预训练的Inception模型，并修改最后一层以匹配你的6个类别
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 6)  # 修改为6个输出类别
model.load_state_dict(torch.load("/root/NEU-FewShot/models/resnet50_expansion_1shot-finetuning_image-to-text.pth"))

# 将模型移动到GPU（如果有的话）
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_dataset = ImageFolder(root="/root/NEU-FewShot/NEU-data/testing-dataset/", transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
acc = 0
for i, (inputs, labels) in enumerate(test_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, pred_result = torch.max(outputs.data, 1)
    acc += torch.sum(pred_result.data == labels.data)
print(acc.item()/600)