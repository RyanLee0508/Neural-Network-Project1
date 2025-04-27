from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key][...] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key][...] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}

        # Initialize velocity for each parameter in the model
        for layer in self.model.layers:
            if layer.optimizable:  # 检查层是否可优化
                self.velocity[layer] = {}
                for key in layer.params.keys():
                    self.velocity[layer][key] = np.zeros_like(layer.params[key])
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    # 批量更新
                    self.velocity[layer][key] = self.mu * self.velocity[layer][key] - self.init_lr * layer.grads[key]
                    layer.params[key] += self.velocity[layer][key] - self.init_lr * layer.weight_decay_lambda * layer.params[key]
                    
class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

        # 为模型中的每个可优化参数初始化一阶矩估计和二阶矩估计
        for layer in self.model.layers:
            if layer.optimizable:
                self.m[layer] = {}
                self.v[layer] = {}
                for key in layer.params.keys():
                    self.m[layer][key] = np.zeros_like(layer.params[key])
                    self.v[layer][key] = np.zeros_like(layer.params[key])

    def step(self):
        self.t += 1
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    # 计算一阶矩估计
                    self.m[layer][key] = self.beta1 * self.m[layer][key] + (1 - self.beta1) * layer.grads[key]
                    # 计算二阶矩估计
                    self.v[layer][key] = self.beta2 * self.v[layer][key] + (1 - self.beta2) * (layer.grads[key] ** 2)
                    # 偏差修正
                    m_hat = self.m[layer][key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[layer][key] / (1 - self.beta2 ** self.t)
                    # 更新参数
                    update = self.init_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    layer.params[key] -= update - self.init_lr * layer.weight_decay_lambda * layer.params[key]
    