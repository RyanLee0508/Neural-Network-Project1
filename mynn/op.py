from abc import abstractmethod
import numpy as np
import cupy as cp
from cupy.lib.stride_tricks import as_strided

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
    def __init__(self, in_dim, out_dim, initialize_method=cp.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return cp.dot(X, self.W) + self.b

    def backward(self, grad : cp.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        self.grads['W'] = cp.dot(self.input.T, grad) / batch_size
        self.grads['b'] = cp.sum(grad, axis=0, keepdims=True) / batch_size
        if self.weight_decay:
            self.grads['W'] -= self.W * self.weight_decay_lambda
        
        return cp.dot(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, initialize_method=cp.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))
        self.b = initialize_method(size=(out_channels, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        self.input = X
        batch_size, C_in, in_height, in_width = X.shape
        kernal_height, kernal_width = self.kernel_size
        s = self.stride
        p = self.padding

        #添加zero-padding
        if p > 0:
            X = cp.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
        
        #卷积核大小的递归公式
        out_height = (in_height - kernal_height + 2 * p) // s + 1
        out_width = (in_width - kernal_width + 2 * p) // s + 1

        # 构造滑动窗口
        shape = (batch_size, C_in, out_height, out_width, kernal_height, kernal_width)
        strides = (
            X.strides[0],
            X.strides[1],
            s * X.strides[2],
            s * X.strides[3],
            X.strides[2],
            X.strides[3],
        )
        windows = as_strided(X, shape=shape, strides=strides) 
        self.windows = windows

        # 利用 einsum 高效计算卷积
        output = cp.einsum('bihwuv,oiuv->bohw', windows, self.W)  
        output += self.b.reshape(1, -1, 1, 1)
        return output


    def backward(self, grad_output):
        batch_size, C_in, in_height, in_width = self.input.shape
        kernal_height, kernal_width = self.kernel_size
        s = self.stride
        p = self.padding
        out_height, out_width = grad_output.shape[2:]

        #利用enisum高效计算梯度
        dW = cp.einsum('bihwuv,bohw->oiuv', self.windows, grad_output)
        db = grad_output.sum(axis=(0, 2, 3), keepdims=True).reshape(self.b.shape)


        # 反卷积将梯度传回输入
        dX_padded = cp.zeros((batch_size, C_in, in_height + 2 * p, in_width + 2 * p))
        flipped_W = self.W[:, :, ::-1, ::-1]  
        for i in range(out_height):
            for j in range(out_width):
                h_start, w_start = i * s, j * s
                grad_slice = grad_output[:, :, i, j]  
                dX_padded[:, :, h_start:h_start+kernal_height, w_start:w_start+kernal_width] += cp.einsum('bo,oiuv->biuv', grad_slice, flipped_W)

        dX = dX_padded[:, :, p:in_height + p, p:in_width + p] if p > 0 else dX_padded
        self.grads['W'] = dW
        self.grads['b'] = db
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
        output = cp.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = cp.where(self.input < 0, 0, grads)
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
        probs = cp.clip(self.predicts, 1e-15, 1-1e-15)
        
        # 转换 labels 为 one-hot encoding
        one_hot_labels = cp.zeros_like(probs)
        one_hot_labels[cp.arange(len(labels)), labels] = 1
        
        # 计算 cross-entropy loss
        loss = -cp.sum(one_hot_labels * cp.log(probs)) / len(labels)
        return loss
    
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = len(self.labels)
        one_hot_labels = cp.zeros_like(self.predicts)
        one_hot_labels[cp.arange(batch_size), self.labels] = 1
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
                    reg_loss += cp.sum(layer.params[key] ** 2)
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
    x_max = cp.max(X, axis=1, keepdims=True)
    x_exp = cp.exp(X - x_max)
    partition = cp.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class MaxPooling2D(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.optimizable = False  # 不可优化
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch_size, channels, input_height, input_width = X.shape
        kernel_height, kernel_width = self.kernel_size
        stride = self.stride

        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1

        # 计算窗口
        shape = (batch_size, channels, output_height, output_width, kernel_height, kernel_width)
        strides = (
            X.strides[0],  
            X.strides[1],  
            stride * X.strides[2],  
            stride * X.strides[3],  
            X.strides[2],  
            X.strides[3],  
        )
        windows = as_strided(X, shape=shape, strides=strides)
            
        # 前向传播最大值
        output = cp.max(windows, axis=(4, 5))  
            
        # 记录最大值的位置，用于反向传播
        max_mask = (windows == output[..., None, None])
        self.max_indices = max_mask.astype(cp.uint8)
        return output

    def backward(self, grads):
        batch_size, channels, output_height, output_width = grads.shape
        kernel_height, kernel_width = self.kernel_size
        stride = self.stride
        input_height, input_width = self.input.shape[2], self.input.shape[3]
        grad_input = cp.zeros_like(self.input)
        shape = (batch_size, channels, output_height, output_width, kernel_height, kernel_width)
        strides = (
            grad_input.strides[0],
            grad_input.strides[1],
            stride * grad_input.strides[2],
            stride * grad_input.strides[3],
            grad_input.strides[2],
            grad_input.strides[3],
        )
        grad_windows = as_strided(grad_input, shape=shape, strides=strides)
            
        # 反向传播最大值
        grad_windows += (self.max_indices * grads[..., None, None])
        return grad_input
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False  # 不可优化
        self.input_shape = None
        self.params = {}  
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)