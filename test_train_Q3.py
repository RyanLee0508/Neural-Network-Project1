# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
import os
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 定义不同的正则化方法
regularization_methods = [
    ("L2 Regularization", [1e-4], None, None),
    ("Dropout", None, 0.5, None),
    ("Early Stopping", None, None, 10)
]

# 存储每个方法的验证准确率
results = {}

for name, lambda_list, dropout_prob, early_stopping_patience in regularization_methods:
    print(f"Training model with {name}")
    linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', lambda_list, dropout_prob)
    optimizer = nn.optimizer.MomentGD(init_lr=0.5, model=linear_model, mu=0.9)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max() + 1)

    runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, early_stopping_patience=early_stopping_patience)
    save_dir = f'./best_models/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=4, log_iters=100, save_dir=save_dir)

    # 记录验证集上的最佳准确率
    results[name] = runner.best_score

# 打印所有方法的验证集准确率
for name, accuracy in results.items():
    print(f"{name}: Validation Accuracy = {accuracy:.4f}")