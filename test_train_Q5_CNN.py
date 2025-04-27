import cupy as cp
import os
import mynn as nn
import gzip
from struct import unpack

# 加载数据
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num, 1, rows, cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = cp.frombuffer(f.read(), dtype=cp.uint8)

# 划分训练集和验证集
cp.random.seed(42)
idx = cp.random.permutation(cp.arange(num))
edge_1 = 5000
edge_2 = 10000
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[edge_1:edge_2]  
valid_labs = train_labs[edge_1:edge_2]
train_imgs = train_imgs[:edge_1]  
train_labs = train_labs[:edge_1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

# 检查 GPU 是否可用
if cp.cuda.is_available():
    print("GPU is available.")
    print(cp.cuda.runtime.getDeviceCount(), "GPU(s) detected.")
else:
    print("GPU is not available. Using CPU instead.")

# # 确保数据在 GPU 上
# train_imgs = cp.asarray(train_imgs)
# train_labs = cp.asarray(train_labs)
# valid_imgs = cp.asarray(valid_imgs)
# valid_labs = cp.asarray(valid_labs)

# 确保模型在 GPU 上
def to_device(model, device=0):
    for layer in model.layers:
        if hasattr(layer, 'W'):
            layer.W = cp.asarray(layer.W)
        if hasattr(layer, 'b'):
            layer.b = cp.asarray(layer.b)
            
# 创建CNN模型
cnn_model = nn.models.Model_CNN()

# 调用函数将模型参数移动到 GPU
to_device(cnn_model)

# 创建优化器和学习率调度器
optimizer = nn.optimizer.Adam(init_lr=0.007, model=cnn_model)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=32,gamma=0.7)

# 创建损失函数
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

# 创建RunnerM实例
runner = nn.runner.RunnerM(cnn_model, optimizer, metric=nn.metric.accuracy, loss_fn=loss_fn, batch_size=128, scheduler=scheduler)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=20, log_iters=1, save_dir='./best_models/CNN')