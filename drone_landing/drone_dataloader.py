import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as F

folder_path = '../data_train/images'
label_path = '../data_train/labels'

data_files = os.listdir(folder_path)
label_files = os.listdir(label_path)

# fixed_size = [540, 540]

transform = transforms.Compose([
    # transforms.Resize(640),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

data = []
for file_name in data_files:
    image_path = os.path.join(folder_path, file_name)
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = np.transpose(image_tensor, (1, 2, 0))
    data.append(image_tensor)
    # box [x, y, h, w] normalized
    box = []

bbox = []
for file_name in label_files:
    with open(os.path.join(label_path, file_name)) as label_file:
        for line in label_file:
            box = [float(i) for i in line.split(' ')]
            bbox.append(box)

print('data len: ', len(data_files))
print('label len: ', len(label_files))
print(data[1].shape)
plt.imshow(data[1])
print(bbox[1])
plt.show()
