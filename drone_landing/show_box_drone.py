import matplotlib.pyplot as plt
import cv2

data_dir = '../data_train/images/img_train_49.jpg'
label_dir = '../data_train/labels/img_train_49.txt'

sample_image = cv2.imread(data_dir)
print(sample_image.shape)

box = []
with open(label_dir) as label_file:
    for line in label_file:
        box = [float(i) for i in line.split(' ')]
print(box)

h = int(box[3] * sample_image.shape[0])
w = int(box[4] * sample_image.shape[1])
print(h, w)

pt1 = (int(box[1] * sample_image.shape[1] - h / 2), int(box[2] * sample_image.shape[0] - w / 2))
pt2 = (int(box[1] * sample_image.shape[1] + h / 2), int(box[2] * sample_image.shape[0] + w / 2))
print(pt1, pt2)

cv2.rectangle(sample_image, pt1, pt2, thickness=2, color=(0, 0, 255))
img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
