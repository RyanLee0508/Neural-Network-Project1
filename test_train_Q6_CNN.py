import cupy as cp
import numpy as np
import os
import mynn as nn
import gzip
from struct import unpack
from scipy.ndimage import rotate, shift

# 数据增强函数
def data_augmentation(image, label, num_augmentations):
    """
    对输入图像进行数据增强。
    :param image: 输入图像，形状为 [batch_size, channels, height, width]
    :param label: 输入标签，形状为 [batch_size, ]
    :param num_augmentations: 每张图像生成的增强样本数量
    :return: 增强后的图像和标签
    """
    batch_size, channels, height, width = image.shape
    augmented_images = []
    augmented_labels = []

    for i in range(batch_size):
        img = image[i]
        lbl = label[i]

        # 原始图像
        augmented_images.append(img)
        augmented_labels.append(lbl)

        # 数据增强
        for _ in range(num_augmentations):
            # 随机选择增强方式的索引
            aug_index = np.random.choice([0, 1, 2])  # 使用 numpy 而不是 cupy
            aug_type = ['translate', 'rotate', 'noise'][aug_index]

            if aug_type == 'translate':
                # 随机平移
                shift_x = np.random.randint(-3, 4)
                shift_y = np.random.randint(-3, 4)
                img_np = img.get()  # 将 CuPy 数组转换为 NumPy 数组
                img_aug_np = shift(img_np, (0, shift_x, shift_y), mode='nearest')
                img_aug = cp.asarray(img_aug_np)  # 将 NumPy 数组转换回 CuPy 数组
            elif aug_type == 'rotate':
                # 随机旋转
                angle = np.random.uniform(-10, 10)
                img_np = img.get()  # 将 CuPy 数组转换为 NumPy 数组
                img_aug_np = rotate(img_np, angle, reshape=False, mode='nearest')
                img_aug = cp.asarray(img_aug_np)  # 将 NumPy 数组转换回 CuPy 数组
            elif aug_type == 'noise':
                # 添加高斯噪声
                noise = np.random.normal(0, 0.1, img.shape)
                img_np = img.get()  # 将 CuPy 数组转换为 NumPy 数组
                img_aug_np = img_np + noise
                img_aug_np = np.clip(img_aug_np, 0, 1)  # 限制在 [0, 1] 范围内
                img_aug = cp.asarray(img_aug_np)  # 将 NumPy 数组转换回 CuPy 数组

            augmented_images.append(img_aug)
            augmented_labels.append(lbl)

    augmented_images = cp.array(augmented_images)
    augmented_labels = cp.array(augmented_labels)
    return augmented_images, augmented_labels

# 加载数据
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num, 1, rows, cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = cp.frombuffer(f.read(), dtype=cp.uint8)

# 数据增强
train_imgs_aug, train_labs_aug = data_augmentation(train_imgs, train_labs, num_augmentations=1)

# 划分训练集和验证集
cp.random.seed(42)
idx = cp.random.permutation(cp.arange(train_imgs_aug.shape[0]))
train_imgs_aug = train_imgs_aug[idx]
train_labs_aug = train_labs_aug[idx]

edge_1 = 2000
edge_2 = 4000
valid_imgs = train_imgs_aug[edge_1:edge_2]  
valid_labs = train_labs_aug[edge_1:edge_2]
train_imgs_aug = train_imgs_aug[:edge_1]  
train_labs_aug = train_labs_aug[:edge_1]
train_imgs_aug = train_imgs_aug / 255.0
valid_imgs = valid_imgs / 255.0

# 检查 GPU 是否可用
if cp.cuda.is_available():
    print("GPU is available.")
    print(cp.cuda.runtime.getDeviceCount(), "GPU(s) detected.")
else:
    print("GPU is not available. Using CPU instead.")

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
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=32, gamma=0.7)

# 创建损失函数
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

# 创建RunnerM实例
runner = nn.runner.RunnerM(cnn_model, optimizer, metric=nn.metric.accuracy, loss_fn=loss_fn, batch_size=128, scheduler=scheduler)
runner.train([train_imgs_aug, train_labs_aug], [valid_imgs, valid_labs], num_epochs=30, log_iters=1, save_dir='./best_models/CNN_aug')
