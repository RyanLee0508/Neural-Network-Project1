from .op import *
import pickle
import numpy as np  
import cupy as cp  


class Model_MLP(Layer):
    """
    A model with linear layer   s. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_prob=None):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_prob = dropout_prob    # 支持dropout
        self.training = True

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i] if i < len(lambda_list) else lambda_list[-1]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if dropout_prob is not None:
                        self.layers.append(Dropout(drop_prob=dropout_prob))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                outputs = layer(outputs, training=self.training)  # 传递 training 参数
            else:
                outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        super().__init__()
        self.conv1 = conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1.W = cp.asarray(self.conv1.W)  # 确保权重是 cupy 数组
        self.conv1.b = cp.asarray(self.conv1.b)  # 确保偏置是 cupy 数组
        self.relu1 = ReLU()
        self.pool1 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        self.conv2 = conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2.W = cp.asarray(self.conv2.W)  # 确保权重是 cupy 数组
        self.conv2.b = cp.asarray(self.conv2.b)  # 确保偏置是 cupy 数组
        self.relu2 = ReLU()
        self.pool2 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        self.flatten = Flatten()
        self.fc1 = Linear(in_dim=64 * 7 * 7, out_dim=128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_dim=128, out_dim=10)

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.fc1, self.relu3, self.fc2
        ]
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.pool2(X)
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.relu3(X)
        X = self.fc2(X)
        return X

    def backward(self, loss_grad):
        grads = loss_grad
        grads = self.fc2.backward(grads)
        grads = self.relu3.backward(grads)
        grads = self.fc1.backward(grads)
        grads = self.flatten.backward(grads)
        grads = self.pool2.backward(grads)
        grads = self.relu2.backward(grads)
        grads = self.conv2.backward(grads)
        grads = self.pool1.backward(grads)
        grads = self.relu1.backward(grads)
        grads = self.conv1.backward(grads)
        return grads

    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        layers = [self.conv1, self.relu1, self.pool1, self.conv2, self.relu2, self.pool2, self.flatten, self.fc1, self.relu3, self.fc2]
        param_index = 0
        
        for layer in layers:
            if layer.optimizable:
                layer.W = param_list[param_index]['W']
                layer.b = param_list[param_index]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[param_index]['weight_decay']
                layer.weight_decay_lambda = param_list[param_index]['weight_decay_lambda']
                param_index += 1
            
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        
    def train(self):
        self.training = True

    def eval(self):
        self.training = False
        
        
class Dropout(Layer):
    def __init__(self, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.optimizable = False

    def __call__(self, X, training=False):
        return self.forward(X, training=training)

    def forward(self, X, training=False):
        if training:  # 只在训练时应用Dropout
            self.mask = cp.random.rand(*X.shape) > self.drop_prob
            return X * self.mask / (1 - self.drop_prob)
        else:
            return X

    def backward(self, grads, training=False):
        if training:
            return grads * self.mask / (1 - self.drop_prob)
        else:
            return grads
        
