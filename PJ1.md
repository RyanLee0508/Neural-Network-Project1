# 神经网络与深度学习——PJ1

#### <u>**22307130305				李林翰**</u>



## README基础部分

### 线性模型

按照README文件中的要求进行了代码填充。

使用 *SGD* 优化器的结果：

![image-20250416161029076](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250416161029076.png)

后续迭代结果相差不大，由此可见误差太大，分数太低，模型可能在乱猜。

使用自己编写的 *MomentGD* 优化器结果：

![image-20250416161133472](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250416161133472.png)

误差大大减少，且分数在**九十分**左右。

总共训练迭代结束后的结果：

![image-20250416162537755](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250416162537755.png)

最佳准确度为**0.90940**。

训练迭代**loss**与**score**的变化图：

![Figure_1](C:\Users\LILINHAN\Desktop\pro_codes\Figure_1.png)

从图中可以看出，训练损失和准确率持续改善，而验证损失和准确率在一定迭代次数后趋于平稳，表明模型在训练集上持续学习，但在验证集上可能开始出现过拟合的迹象。



最后测试结果为：

![image-20250416164514070](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250416164514070.png)

准确率为**0.9104**

- **GitHub总代码链接**：[https://github.com/RyanLee0508/Neural-Network-Project1.git](https://github.com/yourusername/Neural-Network-Project)
- **数据集链接**（接下来使用的数据集都为此数据集链接，不再重复引出）：
  - MNIST训练图像：[https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/dataset/MNIST/train-images-idx3-ubyte.gz](https://github.com/yourusername/Neural-Network-Project/blob/main/dataset/train-images-idx3-ubyte.gz)
  - MNIST训练标签：[https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/dataset/MNIST/train-labels-idx1-ubyte.gz](https://github.com/yourusername/Neural-Network-Project/blob/main/dataset/train-labels-idx1-ubyte.gz)
- **训练好的模型权重链接**：
  - 最佳模型权重：[https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/saved_models/best_model.pickle](https://github.com/yourusername/Neural-Network-Project/blob/main/models/best_model.pickle)



## PDF进阶部分

### Q1:

本题我们尝试了不同隐藏层数目（一层/两层）以及隐藏层中不同神经元数（300/600）的试验测试，结果如下：

![image-20250418145616603](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250418145616603.png)

解读如下：

- **模型容量与准确率**：从结果中可以看出，并不是隐藏层神经元数量越多，模型的性能就越好。相反，具有两个隐藏层（600和300个神经元）的模型表现最差。这可能是因为模型过于复杂，导致在有限的训练数据上出现了过拟合现象，即模型在训练集上表现良好，但在验证集上泛化能力差。

- **简化模型的优势**：可以发现相同隐藏层数的情况下，具有较少神经元的模型（如MLP_[300, 10]）在验证集上的表现优于更复杂的模型，且与具有一层600神经元的隐藏层的模型表现类似。这表明在这种情况下，简化模型可以提高模型的泛化能力，减少过拟合的风险。

  

**Q1的训练代码链接：**https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/test_train_Q1.py

**训练好的模型权重链接**：

- MLP[300, 10]权重：https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/best_models/MLP_%5B300%2C%2010%5D/best_model.pickle

- MLP[600, 10]权重：  https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/best_models/MLP_%5B600%2C%2010%5D/best_model.pickle
- MLP[600, 300, 10]权重：  https://github.com/RyanLee0508/Neural-Network-Project1/blob/main/best_models/MLP_%5B600%2C%20300%2C%2010%5D/best_model.pickle



### Q2：

本题我们尝试了不同学习率（步长）的试验测试，结果如下：

![image-20250418155002334](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250418155002334.png)

可发现这三组测试结果中，随着步长增大，准确度也随之上升，但理论步长过大可能导致损失函数震荡、训练发散或模型欠拟合。所以我们又增加了几组步长测试：

Learning Rate: 0.2, Validation Accuracy = 0.9335

Learning Rate: 0.5, Validation Accuracy = 0.9463

Learning Rate: 0.7, Validation Accuracy = 0.9436

Learning Rate: 0.9, Validation Accuracy = 0.9386

可以发现试验是有最优步长的，大概为0.5左右，且符合步长对准确性的影响。

**Q2的训练代码链接：**https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/test_train_Q2.py

**训练好的模型权重链接**：https://github.com/RyanLee0508/Neural-Network-Project1/tree/master/best_models（含有上述所有学习率的模型权重）



### Q3：

本题我们尝试了不同正则化方法的试验测试（学习率为Q2得到的最大准确度值学习率0.5），结果如下：

![image-20250420174823186](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250420174823186.png)

发现正则化后的分数（例如准确率、损失值等）和正则化前差不多，而正则化的主要目的是减少模型的复杂性，从而防止过拟合。如果正则化前后分数没有显著变化，可能意味着模型在正则化之前已经没有过拟合，或者过拟合的程度非常轻微。

**Q3的训练代码链接：**https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/test_train_Q3.py

**训练好的模型权重链接**：

- L2 Regularization权重：https://github.com/RyanLee0508/Neural-Network-Project1/tree/master/best_models/L2%20Regularization
- Dropout权重：https://github.com/RyanLee0508/Neural-Network-Project1/tree/master/best_models/Dropout
- Early Stopping权重：https://github.com/RyanLee0508/Neural-Network-Project1/tree/master/best_models/Early%20Stopping



### Q4：

从一开始我们就补全且使用`MultiCrossEntropyLoss`，而Softmax层已经在`MultiCrossEntropyLoss`类中隐式地实现了（通过`self.predicts = softmax(predicts)`）。故此题我们认为重心为显示“10 outputs can be interpreted as probabilities of each class”的结果。

调整runner文件中的代码，使每次迭代的预测概率都打印出来，我们随机截取了一段结果：

![image-20250421110438677](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250421110438677.png)

可以发现，第394个样本，我们训练的模型认为它是9的概率为1，为其他数字的概率趋近于0；第395个样本，我们训练的模型认为它是8的概率为1，为其他数字的概率趋近于0；第396个样本，我们训练的模型认为它是0的概率为1，为其他数字的概率趋近于0……

**含交叉熵以及softmax代码的`op`文件链接：**https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/mynn/op.py

**为了显示概率而更改的`runner`文件链接：**https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/mynn/runner.py

更改部分为被注释的代码处（因为全部显示概率对于后续实验结果显示过于繁琐，选择将其在后续实验中注释，防止文件被覆盖导致无法查看更改，此处直接贴出代码截屏）：![image-20250421140225943](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250421140225943.png)

### Q5:

因为本台电脑CPU三个小时才能训练出一次迭代：

![image-20250425094230329](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250425094230329.png)

甚至采用tqdm进度条后，发现总共需要50多个小时，速度太慢，并且运行过程中，不到百分之一，内存就爆炸了，所以需要对代码进行大改与更新：

- 采用cupy，将程序切换至电脑的GPU上运行。
- 将优化器增添Adam优化器，Adam优化器优于已有的SGD与MomentGD，可以加速计算。
- 优化循环，在CNN训练中，原代码常有大量重复for循环，我们采用cupy来加速计算。
- 减少MNIST数据集的训练量，因为总共数字只有10个，60000个样本过多，并且电脑训练时会崩溃，所以我们降低数据量。

下图为训练数据量为1000的结果：

![image-20250427153219313](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250427153219313.png)

下图为训练数据量为5000的结果：

![image-20250427153609807](C:\Users\LILINHAN\AppData\Roaming\Typora\typora-user-images\image-20250427153609807.png)

可以发现准确度为**百分之九十**左右。

**Q5的训练代码链接：**https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/test_train_Q5_CNN.py

##### 更改的文件链接：

- https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/mynn/models.py
- https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/mynn/op.py
- https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/mynn/optimizer.py
- https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/mynn/runner.py

**训练好的模型权重链接**：https://github.com/RyanLee0508/Neural-Network-Project1/blob/master/best_models/CNN/best_model.pickle

### Q6：

此题我们构建了三种变换（平移、旋转、高斯噪声）来制造更多训练样本，结果如下：

![屏幕截图 2025-04-28 133300](C:\Users\LILINHAN\Pictures\Screenshots\屏幕截图 2025-04-28 133300.png)

可见，此题准确度为**百分之八十二**低于**Q5**问题的准确度，我们可以发现经过变换后的图像可能会造成特征混乱，例如有些手写数字4可能旋转过后会更像6，导致学习难度增大，准确度变低一些。

