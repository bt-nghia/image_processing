import torch
from torchsummary import summary

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
summary(model)
print(model)