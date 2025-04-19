from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        self.grads['W'] = np.dot(self.input.T, grad) / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        if self.weight_decay:
            self.grads['W'] -= self.W * self.weight_decay_lambda
        
        return np.dot(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)  # 检验是否为元组输入
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))
        self.b = initialize_method(size=(out_channels, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_c, in_h, in_w = X.shape
        k_h, k_w = self.kernel_size
        out_h = (in_h + 2 * self.padding - k_h) // self.stride + 1  # 卷积层输出特征图的尺寸公式
        out_w = (in_w + 2 * self.padding - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        # if self.padding > 0:
        #     X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride  # 计算卷积核在图像中的起始位置
                        w_start = j * self.stride
                        patch = X[b, :, h_start:h_start + k_h, w_start:w_start + k_w]  # 获取当前卷积核在图像中的位置
                        output[b, oc, i, j] = np.sum(patch * self.W[oc]) + self.b[oc]  # 计算卷积核与图像的乘积并加上偏置项

        return output
    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        batch_size, in_c, in_h, in_w = X.shape
        k_h, k_w = self.kernel_size
        out_h, out_w = grads.shape[2], grads.shape[3] 

        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))  # batch_size, in_c, in_h, in_w = X.shape  意为填充后两项
            dX = np.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = X[b, :, h_start:h_start + k_h, w_start:w_start + k_w]

                        dW[oc] += grads[b, oc, i, j] * patch 
                        db[oc] += grads[b, oc, i, j]
                        dX[b, :, h_start:h_start + k_h, w_start:w_start + k_w] += grads[b, oc, i, j] * self.W[oc]  # 计算卷积核对图像的梯度

        if self.padding > 0:
            dX = dX[:, :, self.padding:-self.padding, self.padding:-self.padding]  # 去除填充

        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size
        if self.weight_decay:
            self.grads['W'] -= self.W * self.weight_decay_lambda

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.has_softmax = True
        self.max_classes = max_classes
        self.predicts = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        if self.has_softmax:
            self.predicts = softmax(predicts)
        else:
            self.predicts = predicts

        # 确保数据不为0
        probs = np.clip(self.predicts, 1e-15, 1-1e-15)
        
        # 转换 labels 为 one-hot encoding
        one_hot_labels = np.zeros_like(probs)
        one_hot_labels[np.arange(len(labels)), labels] = 1
        
        # 计算 cross-entropy loss
        loss = -np.sum(one_hot_labels * np.log(probs)) / len(labels)
        return loss
    
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = len(self.labels)
        one_hot_labels = np.zeros_like(self.predicts)
        one_hot_labels[np.arange(batch_size), self.labels] = 1
        self.grads = self.predicts - one_hot_labels
        self.grads /= batch_size
        
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    # 多考虑L2正则项
    def __init__(self, model, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.model = model
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self):
        """
        Calculate the L2 regularization term.
        """
        reg_loss = 0.0
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    reg_loss += np.sum(layer.params[key] ** 2)
        return 0.5 * self.weight_decay_lambda * reg_loss

    def backward(self):
        """
        Update the gradients of the parameters with the L2 regularization term.
        """
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.grads[key] is None:
                        layer.grads[key] = 0.0
                    layer.grads[key] += self.weight_decay_lambda * layer.params[key]

    def clear_grad(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    layer.grads[key] = None
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition