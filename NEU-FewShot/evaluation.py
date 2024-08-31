import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim


def calculate_entropy(probabilities):  # probabilities是基础分类器输出的logits
    # Compute the posterior probability
    entropy_total = -np.sum(probabilities * np.log2(probabilities + 1e-9), axis=1)
    entropy = np.mean(entropy_total)
    return entropy


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # 调整图片大小为299x299
        transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为三通道图片
        transforms.ToTensor(),  # 将PIL图片转换为tensor，并归一化到[0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(image)


# 加载预训练的Inception模型，并修改最后一层以匹配你的6个类别
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 5)  # 修改为6个输出类别
model.load_state_dict(torch.load("/root/NEU-FewShot/models/resnet50_training.pth"))
model.eval()

# 将模型移动到GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class_name_list = ["inclusion", "oil_spot", "punching_hole", "water_spot", "welding_line"]
result_entropy_list = np.zeros((5, 50))
original_entropy_list = np.zeros((5, 50))
ssim_list = np.zeros((5, 50))
fid_dist = np.zeros(5)

for class_idx in range(5):
    original_path = "/root/NEU-FewShot/data/testing-dataset/" + class_name_list[class_idx] + "/"
    result_path = "/root/NEU-FewShot/data/randaugment_expansion-dataset/" + class_name_list[class_idx] + "/"

    fid_dist[class_idx] = fid_score.calculate_fid_given_paths([original_path, result_path],
                                                              batch_size=64, device=device, dims=2048)
    for idx in range(50):
        result_img_path = result_path+os.listdir(result_path)[idx]
        original_img_path = original_path+f'test_{(idx+1):04d}.jpg'

        result_img = Image.open(result_img_path)
        original_img = Image.open(original_img_path)
        ssim_list[class_idx, idx] = ssim(cv2.imread(result_img_path, cv2.IMREAD_GRAYSCALE), cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE))

        result_tensor = transform_image(result_img).unsqueeze(0).to(device)
        original_tensor = transform_image(original_img).unsqueeze(0).to(device)

        result_tensor_logits = F.softmax(model(result_tensor)).cpu().detach().numpy()
        original_tensor_logits = F.softmax(model(original_tensor)).cpu().detach().numpy()

        result_entropy_list[class_idx, idx] = calculate_entropy(result_tensor_logits)
        original_entropy_list[class_idx, idx] = calculate_entropy(original_tensor_logits)

print("IE:")
original_ie_print=""
for ie in original_entropy_list.mean(axis=1):
    original_ie_print += "{:.3f}|".format(ie)
result_ie_print=""
for ie in result_entropy_list.mean(axis=1):
    result_ie_print += "{:.3f}|".format(ie)
print(original_ie_print)
print(result_ie_print)
print(result_entropy_list.mean(axis=1).mean())

print("FID")
fid_print = ""
for fid in fid_dist:
    fid_print += "{:.1f}|".format(fid)
print(fid_print, "{:.1f}".format(fid_dist.mean()))

print("SSIM:")
ssim_print = ""
for ssim in ssim_list.mean(axis=1):
    ssim_print += "{:.3f}|".format(ssim)
print(ssim_print)
print(ssim_list.mean(axis=1).mean())

