import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# 固定随机种子


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA的话
    torch.backends.cudnn.deterministic = True  # 保证CUDNN的确定性
    torch.backends.cudnn.benchmark = False

# 1. 载入数据


def load_data():
    X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 将标签转换为张量
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return X_train, X_test, y_train, y_test

# 2. 定义全连接模型


class FCModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FCModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # 手动初始化模型权重
        self._initialize_weights()

    def forward(self, X):
        X = self.relu(self.linear1(X))
        X = self.relu(self.linear2(X))
        X = self.linear3(X)
        return X  # 输出 logits（未激活的）

    def _initialize_weights(self):
        # 使用固定的初始化方法，确保每次运行时权重相同
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.5)  # 所有权重初始化为0.5
                nn.init.constant_(m.bias, 0.1)  # 所有偏置初始化为0.1

# 3. 训练函数


def train(model, X_train, y_train, epochs=10, batch_size=32, shuffle=False, lr=0.1):
    criterion = nn.BCEWithLogitsLoss()  # 包括sigmoid和cross-entropy
    optimizer = optim.SGD(model.parameters(), lr=lr)

    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = int(np.ceil(num_samples / batch_size))

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # 前向传播
            out = model(X_batch)

            # 计算损失
            loss = criterion(out, y_batch)
            epoch_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新参数
            optimizer.step()

        # 平均损失
        epoch_loss /= num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# 4. 测试模型


def test_fc_model():
    # 固定随机种子
    set_seed(42)

    # 载入数据
    X_train, X_test, y_train, y_test = load_data()

    # 初始化模型
    model = FCModel(input_size=X_train.shape[1], hidden_size1=10, hidden_size2=4, output_size=1)

    # 训练模型
    train(model, torch.tensor(X_train, dtype=torch.float32), y_train, epochs=100, batch_size=32, shuffle=False)

    # 简单的预测
    out = model(torch.tensor(X_train, dtype=torch.float32))
    predictions = torch.sigmoid(out).detach().numpy()  # sigmoid转换为概率

    # 计算AUC
    auc = roc_auc_score(y_train.numpy(), predictions)
    print(f"Test AUC: {auc:.4f}")

    # 断言AUC大于阈值
    assert auc > 0.85, f"auc {auc} is below the threshold"

    print("Test passed!")


if __name__ == "__main__":
    # 运行测试
    test_fc_model()
