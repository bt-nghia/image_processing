from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchsummary import summary
import pandas as pd

data_dir = '../data/images/zidane.jpg'
sample_image = cv2.imread(data_dir)

print(sample_image.shape)

yolov5s_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

out = yolov5s_model(sample_image)
print('out: ', out.xyxy[0])

print(out.xyxy[0][:, 0])
print(out.xyxy[0][:, 1])
print(out.xyxy[0][:, 2])
print(out.xyxy[0][:, 3])

start_point_x = out.xyxy[0][:, 0]
start_point_y = out.xyxy[0][:, 1]
end_point_x = out.xyxy[0][:, 2]
end_point_y = out.xyxy[0][:, 3]

for i in range(3):
    cv2.rectangle(sample_image, (int(start_point_x[i].item()),
                                 int(start_point_y[i].item())),
                                (int(end_point_x[i].item()),
                                 int(end_point_y[i].item())),
                  color=(255, 0, 0), thickness=2)

print(sample_image.shape)
cv2.imwrite('detect.jpg', sample_image)