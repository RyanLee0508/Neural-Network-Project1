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

- **GitHub代码链接**：[https://github.com/RyanLee0508/Neural-Network-Project1.git](https://github.com/yourusername/Neural-Network-Project)
- **数据集链接**：
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

可以发
