# HW1: 从零实现 MLP 完成 Fashion-MNIST 分类

本作业使用 `NumPy` 从零实现一个三层 MLP 分类器，在 `Fashion-MNIST` 数据集上完成训练、验证、测试、超参数搜索、权重可视化和错误分析。

当前代码不依赖 `PyTorch`、`TensorFlow`、`JAX` 等自动微分框架，按作业要求使用手写前向传播、反向传播和 `SGD` 优化。

## 1. 环境说明


```powershell
pip install -r .\requirements.txt
```


## 2. 目录结构

```text
MLP/
├─ README.md
├─ requirements.txt
└─ mlp/
   ├─ core/
   │  ├─ layers.py
   │  ├─ loss.py
   │  ├─ model.py
   │  └─ optim.py
   ├─ utils/
   │  ├─ data_loader.py
   │  └─ metrics.py
   ├─ data/          # 本地放置 Fashion-MNIST 原始 .gz 数据
   ├─ models/        # 本地训练后生成的权重与日志，不提交到 git
   ├─ figures/       # 本地生成的图表，不提交到 git
   ├─ train.py
   ├─ test.py
   ├─ search.py
   ├─ plot_curves.py
   ├─ visualize_weights.py
   └─ error_analysis.py
```

## 3. 代码功能

- `core/model.py`: MLP 模型定义
- `core/layers.py`: 线性层、ReLU、Sigmoid
- `core/loss.py`: Cross-Entropy Loss
- `core/optim.py`: SGD 和 L2 正则化
- `utils/data_loader.py`: Fashion-MNIST 数据读取与 batch 划分
- `utils/metrics.py`: Accuracy 和 Confusion Matrix
- `train.py`: 训练模型并按验证集准确率保存最优权重
- `test.py`: 加载最优模型，在测试集上评估并输出混淆矩阵
- `search.py`: 网格搜索超参数，并将最优模型复制为 `best_model.pkl`
- `plot_curves.py`: 绘制训练/验证损失曲线与验证集准确率曲线
- `visualize_weights.py`: 可视化第一层隐藏层权重
- `error_analysis.py`: 展示测试集中的错误分类样本

## 4. 运行方法

下面的命令默认在仓库根目录下执行。

### 4.0 准备数据集


- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

### 4.1 训练模型

```powershell
python .\mlp\train.py
```

作用：

- 读取 `mlp/data/` 下的 Fashion-MNIST 数据
- 划分训练集和验证集
- 训练 MLP
- 按验证集最优准确率保存模型
- 保存训练历史

训练完成后会生成：

- `mlp/models/best_model.pkl`
- `mlp/models/history.pkl`

### 4.2 测试模型

```powershell
python .\mlp\test.py
```

作用：

- 加载 `mlp/models/best_model.pkl`
- 在测试集上输出准确率
- 打印混淆矩阵

### 4.3 绘制 Loss/Accuracy 曲线

```powershell
python .\mlp\plot_curves.py
```

依赖文件：

- `mlp/models/history.pkl`

输出文件：

- `mlp/figures/loss.png`
- `mlp/figures/acc.png`

### 4.4 第一层权重可视化

```powershell
python .\mlp\visualize_weights.py
```

依赖文件：

- `mlp/models/best_model.pkl`

输出文件：

- `mlp/figures/first_layer_weights.png`

### 4.5 错误分析

```powershell
python .\mlp\error_analysis.py
```

依赖文件：

- `mlp/models/best_model.pkl`

输出文件：

- `mlp/figures/error_analysis.png`

### 4.6 超参数搜索

```powershell
python .\mlp\search.py
```

作用：

- 对学习率、隐藏层维度、权重衰减、激活函数做网格搜索
- 记录每组配置的验证集表现
- 将最优模型复制为 `mlp/models/best_model.pkl`

输出文件：

- `mlp/models/search.csv`
- 若干按配置命名的模型文件
- `mlp/models/best_model.pkl`

## 5. 执行顺序



```powershell
python .\mlp\train.py
python .\mlp\test.py
python .\mlp\plot_curves.py
python .\mlp\visualize_weights.py
python .\mlp\error_analysis.py
python .\mlp\search.py
```

## 6. 默认训练配置

`train.py` 中默认配置如下：

```python
{
    "epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
    "hidden_dim": 128,
    "activation": "relu",
    "val_ratio": 0.2,
    "seed": 42,
    "lr_decay_every": 5,
    "lr_decay_gamma": 0.5,
}
```




