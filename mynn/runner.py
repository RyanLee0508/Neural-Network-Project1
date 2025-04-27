import cupy as cp
import os
import mynn as nn
import numpy as np
from tqdm import tqdm

class RunnerM():  # 支持提前停止
    """
    This is an example to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, early_stopping_patience=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    
        
    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = cp.random.permutation(cp.arange(X.shape[0]))
            X = X[idx]
            y = y[idx]

            self.model.train()  # 设置为训练模式

            # 训练
            epoch_loss = 0.0
            epoch_score = 0.0
            epoch_steps = int(cp.ceil(X.shape[0] / self.batch_size))

            with tqdm(total=epoch_steps, desc=f"EPOCH: {epoch+1:02}/{num_epochs}", ncols=100) as pbar:
                for iteration in range(epoch_steps):
                    # batch training
                    start_idx = iteration * self.batch_size
                    end_idx = min((iteration + 1) * self.batch_size, X.shape[0])
                    train_X = X[start_idx:end_idx]
                    train_y = y[start_idx:end_idx]

                    # forward
                    logits = self.model(train_X)
                    trn_loss = self.loss_fn(logits, train_y)
                    trn_score = self.metric(logits, train_y)

                    # backward
                    self.loss_fn.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # 更新进度条
                    epoch_loss += trn_loss.get()
                    epoch_score += trn_score.get()
                    if (iteration + 1) % log_iters == 0:
                        pbar.set_postfix({
                            "LOSS": f"{epoch_loss / (iteration + 1):.4f}",
                            "ACC": f"{epoch_score / (iteration + 1):.4f}"
                        })
                    pbar.update(1)

            # 计算平均损失和准确率
            epoch_loss /= epoch_steps
            epoch_score /= epoch_steps
            self.train_loss.append(epoch_loss)
            self.train_scores.append(epoch_score)

            # 验证
            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score.get())
            self.dev_loss.append(dev_loss.get())

            # 打印验证结果
            print(f"EPOCH: {epoch+1:02}/{num_epochs} STEP: {epoch_steps}/{epoch_steps} LOSS: {epoch_loss:.4f} ACC: {epoch_score:.4f} VAL-LOSS: {dev_loss.get():.4f} VAL-ACC: {dev_score.get():.4f}")

            # 保存最佳模型
            if dev_score.get() > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"Best accuracy performance has been updated: {best_score:.5f} --> {dev_score.get():.5f}")
                best_score = dev_score.get()
                patience_counter = 0
            else:
                patience_counter += 1

            # 提前停止
            if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    def evaluate(self, data_set):
        self.model.eval()  # 设置为评估模式
        X, y = data_set
        logits = self.model(X)
        # predictions = nn.op.softmax(logits)  # 第四问

        # # 输出每个数字的预测概率
        # for i in range(len(predictions)):
        #     print(f"Sample {i} prediction probabilities: {predictions[i]}")
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)
        