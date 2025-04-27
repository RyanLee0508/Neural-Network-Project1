import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as tn

class CNN(tn.Module):
        def __init__(self):
                super(CNN, self).__init__()
                self.conv = tn.Sequential(
                # [BATCH_SIZE, 1, 28, 28]
                tn.Conv2d(1, 32, 5, 1, 2),
                # [BATCH_SIZE, 32, 28, 28]
                tn.ReLU(),
                tn.MaxPool2d(2),
                # [BATCH_SIZE, 32, 14, 14]
                tn.Conv2d(32, 64, 5, 1, 2),
                # [BATCH_SIZE, 64, 14, 14]
                tn.ReLU(),
                tn.MaxPool2d(2),
                # [BATCH_SIZE, 64, 7, 7]
                )
                self.fc = tn.Linear(64 * 7 * 7, 10)

        def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                y = self.fc(x)
                return y
        
model = CNN()
model.load_model(r'.\best_models\CNN\model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))