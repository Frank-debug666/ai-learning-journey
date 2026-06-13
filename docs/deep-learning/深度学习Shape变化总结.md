# 深度学习中输入输出 Shape 发生改变的情况总结

## 0. 常用符号

| 符号 | 含义 |
|---|---|
| `B` | batch size，一批样本数量 |
| `C` | channel，通道数 |
| `H` | height，图像高度 |
| `W` | width，图像宽度 |
| `F` | feature，特征数量 |
| `T` | sequence length，序列长度 |
| `D` | embedding dimension，词向量维度 |
| `V` | vocabulary size，词表大小 |
| `N` | 样本数量，也常表示 `B*T` |
| `num_classes` | 类别数 |

---

# 1. Shape 改变优先级总表

| 优先级 | 场景 | 常见变化 | 必须掌握程度 |
|---|---|---|---|
| ⭐⭐⭐⭐⭐ | `DataLoader` 加 batch | `[F] → [B, F]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | 图像输入 CNN | `[H,W,C] → [B,C,H,W]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | `Linear` 全连接层 | `[B, in_features] → [B, out_features]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | CNN 卷积层 | `[B,C,H,W] → [B,C_out,H_out,W_out]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | Pooling 池化层 | `[B,C,H,W] → [B,C,H/2,W/2]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | Flatten 展平 | `[B,C,H,W] → [B,C*H*W]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | 分类输出层 | `[B, hidden] → [B, num_classes]` | 必须掌握 |
| ⭐⭐⭐⭐⭐ | 损失函数前 shape 对齐 | `logits` 和 `target` 匹配 | 必须掌握 |
| ⭐⭐⭐⭐ | RNN 输入输出 | `[B,T,D] ↔ [T,B,D]` | 重点掌握 |
| ⭐⭐⭐⭐ | Embedding | `[B,T] → [B,T,D]` | 重点掌握 |
| ⭐⭐⭐⭐ | Transformer Attention | `[B,T,D] → [B,T,D]` | 重点掌握 |
| ⭐⭐⭐ | `unsqueeze/squeeze` | 增加/删除维度 | 常见报错来源 |
| ⭐⭐⭐ | `reshape/view/permute/transpose` | 手动改 shape | 常见报错来源 |
| ⭐⭐⭐ | `concat/stack` | 拼接或堆叠维度 | 项目常用 |
| ⭐⭐ | 上采样/反卷积 | `[B,C,H,W] → [B,C,H*2,W*2]` | 分割/生成任务掌握 |
| ⭐⭐ | 目标检测输出 | 多尺度、多框输出 | 检测方向掌握 |

---

# 2. DataLoader 加 Batch 维度

## 优先级：⭐⭐⭐⭐⭐

单个样本原来可能是：

```python
x.shape = [F]
y.shape = []
```

经过 `DataLoader(batch_size=B)` 后：

```python
x.shape = [B, F]
y.shape = [B]
```

例如表格数据：

```text
单个样本: [10]
一批样本: [64, 10]
```

图像数据：

```text
单张图片: [3, 32, 32]
一批图片: [64, 3, 32, 32]
```

核心记忆：

```text
DataLoader 会自动在最前面加 batch 维度
```

---

# 3. 图像 ToTensor 后的 Shape 改变

## 优先级：⭐⭐⭐⭐⭐

普通图片一般是：

```text
[H, W, C]
```

例如：

```text
[32, 32, 3]
```

经过 `transforms.ToTensor()` 后变成：

```text
[C, H, W]
```

例如：

```text
[3, 32, 32]
```

再经过 DataLoader：

```text
[B, C, H, W]
```

例如：

```text
[64, 3, 32, 32]
```

为什么要这样？

因为 PyTorch 的 CNN 默认输入格式是：

```text
[batch_size, channels, height, width]
```

也就是：

```text
[B, C, H, W]
```

---

# 4. Linear 全连接层的 Shape 改变

## 优先级：⭐⭐⭐⭐⭐

```python
nn.Linear(in_features, out_features)
```

输入：

```text
[B, in_features]
```

输出：

```text
[B, out_features]
```

例如：

```python
nn.Linear(10, 32)
```

shape 变化：

```text
[B, 10] → [B, 32]
```

本质：

```text
每个样本从 10 个特征变成 32 个特征
```

注意：

```python
nn.Linear(10, 32)
```

只改变最后一维：

```text
[..., 10] → [..., 32]
```

例如：

```text
[B, T, 10] → [B, T, 32]
```

---

# 5. CNN 卷积层 Shape 改变

## 优先级：⭐⭐⭐⭐⭐

```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding
)
```

输入：

```text
[B, C_in, H, W]
```

输出：

```text
[B, C_out, H_out, W_out]
```

其中：

```text
C_out = out_channels
```

高度和宽度由公式决定：

```text
H_out = floor((H + 2P - K) / S) + 1
W_out = floor((W + 2P - K) / S) + 1
```

其中：

| 符号 | 含义 |
|---|---|
| `K` | kernel_size |
| `P` | padding |
| `S` | stride |

## 例子 1：保持尺寸不变

```python
nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
```

输入：

```text
[B, 3, 32, 32]
```

输出：

```text
[B, 32, 32, 32]
```

解释：

```text
通道数: 3 → 32
高度宽度: 32 → 32
```

因为：

```text
kernel_size=3, padding=1, stride=1
```

常用于保持图像大小不变。

## 例子 2：尺寸减半

```python
nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
```

输入：

```text
[B, 3, 32, 32]
```

输出：

```text
[B, 32, 16, 16]
```

解释：

```text
stride=2 会让 H 和 W 大约减半
```

---

# 6. Pooling 池化层 Shape 改变

## 优先级：⭐⭐⭐⭐⭐

常见池化：

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

输入：

```text
[B, C, H, W]
```

输出：

```text
[B, C, H/2, W/2]
```

例如：

```text
[B, 32, 32, 32] → [B, 32, 16, 16]
```

注意：

```text
池化层通常不改变通道数 C，只改变 H 和 W
```

---

# 7. Flatten 展平

## 优先级：⭐⭐⭐⭐⭐

CNN 后面接 Linear 之前，通常需要展平。

原始 CNN 输出：

```text
[B, C, H, W]
```

展平后：

```text
[B, C*H*W]
```

代码：

```python
x = x.reshape(x.shape[0], -1)
```

或者：

```python
x = torch.flatten(x, start_dim=1)
```

例如：

```text
[B, 128, 4, 4] → [B, 128*4*4]
```

也就是：

```text
[B, 128, 4, 4] → [B, 2048]
```

核心记忆：

```text
保留 batch 维度，把后面的 C、H、W 合并成一维
```

---

# 8. 分类输出层 Shape 改变

## 优先级：⭐⭐⭐⭐⭐

多分类任务最后一般是：

```python
nn.Linear(hidden_dim, num_classes)
```

输入：

```text
[B, hidden_dim]
```

输出：

```text
[B, num_classes]
```

例如 CIFAR-10 图像分类：

```python
nn.Linear(512, 10)
```

shape：

```text
[B, 512] → [B, 10]
```

含义：

```text
每个样本输出 10 个类别的分数
```

注意：

```text
输出的是 logits，不是概率
```

如果使用：

```python
nn.CrossEntropyLoss()
```

不要手动加 `Softmax`。

---

# 9. 损失函数前后的 Shape 要求

## 优先级：⭐⭐⭐⭐⭐

这是最容易报错的地方。

## 9.1 回归任务：MSELoss / L1Loss / SmoothL1Loss

预测值：

```text
y_pred: [B, 1]
```

真实值：

```text
y_true: [B, 1]
```

二者 shape 必须一致：

```text
[B, 1] 对 [B, 1]
```

或者：

```text
[B] 对 [B]
```

错误例子：

```text
y_pred: [B, 1]
y_true: [B]
```

虽然有时可以广播，但不推荐，容易算错。

推荐改法：

```python
y_true = y_true.reshape(-1, 1)
```

## 9.2 二分类任务：BCEWithLogitsLoss

推荐写法：

```python
loss_f = nn.BCEWithLogitsLoss()
```

模型输出：

```text
y_pred: [B, 1]
```

标签：

```text
y_true: [B, 1]
```

标签类型：

```text
float
```

例如：

```text
y_pred = [[2.1], [-1.3], [0.5]]
y_true = [[1.0], [0.0], [1.0]]
```

注意：

```text
BCEWithLogitsLoss 内部自带 Sigmoid
```

所以模型最后一层不要再加：

```python
torch.sigmoid()
```

## 9.3 多分类任务：CrossEntropyLoss

模型输出：

```text
y_pred: [B, num_classes]
```

标签：

```text
y_true: [B]
```

标签类型：

```text
long
```

例如 10 分类：

```text
y_pred: [64, 10]
y_true: [64]
```

标签内容：

```text
[0, 3, 5, 1, 9, ...]
```

注意：

```text
CrossEntropyLoss 的标签不是 one-hot
```

错误写法：

```text
y_true: [B, num_classes]
```

正确写法：

```text
y_true: [B]
```

---

# 10. Embedding 层 Shape 改变

## 优先级：⭐⭐⭐⭐

常用于 NLP、RNN、Transformer。

```python
nn.Embedding(vocab_size, embedding_dim)
```

输入是单词编号：

```text
[B, T]
```

输出是词向量：

```text
[B, T, D]
```

例如：

```python
nn.Embedding(5000, 256)
```

shape：

```text
[B, T] → [B, T, 256]
```

具体例子：

```text
[32, 20] → [32, 20, 256]
```

解释：

```text
32 个句子
每个句子 20 个词
每个词变成 256 维向量
```

注意：

```text
Embedding 的输入必须是整数编号，dtype 通常是 torch.long
```

---

# 11. RNN / LSTM / GRU 的 Shape 改变

## 优先级：⭐⭐⭐⭐

PyTorch 的 RNN 默认输入格式是：

```text
[T, B, D]
```

如果设置：

```python
batch_first=True
```

输入格式变成：

```text
[B, T, D]
```

## 11.1 默认 RNN：batch_first=False

```python
rnn = nn.RNN(input_size=256, hidden_size=512, num_layers=3)
```

输入：

```text
x: [T, B, 256]
```

初始隐藏状态：

```text
h0: [3, B, 512]
```

输出：

```text
output: [T, B, 512]
hidden: [3, B, 512]
```

## 11.2 batch_first=True

```python
rnn = nn.RNN(
    input_size=256,
    hidden_size=512,
    num_layers=3,
    batch_first=True
)
```

输入：

```text
x: [B, T, 256]
```

输出：

```text
output: [B, T, 512]
hidden: [3, B, 512]
```

注意：

```text
batch_first=True 只影响 input 和 output
不影响 hidden
```

hidden 永远是：

```text
[num_layers, B, hidden_size]
```

---

# 12. RNN 做文本分类时的 Shape

## 优先级：⭐⭐⭐⭐

输入：

```text
x: [B, T]
```

经过 Embedding：

```text
[B, T] → [B, T, D]
```

经过 RNN：

```text
[B, T, D] → [B, T, hidden]
```

如果只拿最后一个时间步做分类：

```python
last_output = output[:, -1, :]
```

shape：

```text
[B, T, hidden] → [B, hidden]
```

接 Linear：

```text
[B, hidden] → [B, num_classes]
```

最终：

```text
y_pred: [B, num_classes]
y_true: [B]
```

适合：

```text
情感分类、文本分类、垃圾邮件分类
```

---

# 13. RNN 做文本生成时的 Shape

## 优先级：⭐⭐⭐⭐

输入：

```text
x: [B, T]
```

Embedding：

```text
[B, T] → [B, T, D]
```

RNN：

```text
[B, T, D] → [B, T, hidden]
```

Linear：

```text
[B, T, hidden] → [B, T, V]
```

为了喂给 `CrossEntropyLoss`，通常展平：

```text
[B, T, V] → [B*T, V]
```

标签也要展平：

```text
[B, T] → [B*T]
```

最终：

```text
y_pred: [B*T, V]
y_true: [B*T]
```

这就是你之前 RNN 代码里：

```python
output.reshape(-1, output.shape[-1])
```

的作用。

---

# 14. Transformer / Attention 的 Shape 改变

## 优先级：⭐⭐⭐⭐

Transformer 常用输入：

```text
[B, T, D]
```

例如：

```text
[32, 128, 768]
```

含义：

```text
32 个句子
每个句子 128 个 token
每个 token 是 768 维向量
```

Self-Attention 一般输入输出 shape 不变：

```text
[B, T, D] → [B, T, D]
```

但是内部会拆成多个头：

```text
[B, T, D] → [B, num_heads, T, head_dim]
```

其中：

```text
D = num_heads * head_dim
```

例如：

```text
D = 768
num_heads = 12
head_dim = 64
```

所以：

```text
[B, 128, 768] → [B, 12, 128, 64]
```

Attention 结束后再合并回来：

```text
[B, 12, 128, 64] → [B, 128, 768]
```

---

# 15. reshape / view 手动改变 Shape

## 优先级：⭐⭐⭐

常见写法：

```python
x = x.reshape(new_shape)
```

或者：

```python
x = x.view(new_shape)
```

例子：

```python
x = x.reshape(x.shape[0], -1)
```

shape：

```text
[B, C, H, W] → [B, C*H*W]
```

注意：

```text
reshape 不会改变数据总数量
```

例如：

```text
[2, 3, 4] 一共有 24 个数
```

可以变成：

```text
[2, 12]
[6, 4]
[24]
```

但不能变成：

```text
[5, 5]
```

因为：

```text
5 * 5 = 25，不等于 24
```

---

# 16. unsqueeze 增加维度

## 优先级：⭐⭐⭐

```python
x = x.unsqueeze(dim)
```

作用：

```text
在指定位置增加一个大小为 1 的维度
```

例子：

```text
[B] → [B, 1]
```

代码：

```python
x = x.unsqueeze(1)
```

常用于修正标签 shape：

```text
y_true: [B] → [B, 1]
```

也常用于单张图片：

```text
[3, 32, 32] → [1, 3, 32, 32]
```

因为模型一般需要 batch 维度。

---

# 17. squeeze 删除维度

## 优先级：⭐⭐⭐

```python
x = x.squeeze(dim)
```

作用：

```text
删除大小为 1 的维度
```

例子：

```text
[B, 1] → [B]
```

代码：

```python
x = x.squeeze(1)
```

注意：

```python
x.squeeze()
```

会删除所有大小为 1 的维度，可能误删 batch 维度。

危险例子：

```text
[1, 1, 32] → [32]
```

推荐：

```python
x.squeeze(1)
```

不要随便写：

```python
x.squeeze()
```

---

# 18. transpose / permute 改变维度顺序

## 优先级：⭐⭐⭐

## 18.1 transpose

```python
x = x.transpose(0, 1)
```

只交换两个维度。

例子：

```text
[B, T, D] → [T, B, D]
```

常用于适配 RNN 默认输入格式。

## 18.2 permute

```python
x = x.permute(0, 3, 1, 2)
```

可以重新排列多个维度。

图像常见例子：

```text
[B, H, W, C] → [B, C, H, W]
```

代码：

```python
x = x.permute(0, 3, 1, 2)
```

用于把普通图片格式转换成 PyTorch CNN 格式。

---

# 19. concat 拼接

## 优先级：⭐⭐⭐

```python
torch.cat([x1, x2], dim=某个维度)
```

作用：

```text
沿着已有维度拼接
```

例子 1：特征拼接

```text
x1: [B, 128]
x2: [B, 64]
```

```python
x = torch.cat([x1, x2], dim=1)
```

输出：

```text
x: [B, 192]
```

例子 2：通道拼接

```text
x1: [B, 32, H, W]
x2: [B, 64, H, W]
```

```python
x = torch.cat([x1, x2], dim=1)
```

输出：

```text
x: [B, 96, H, W]
```

注意：

```text
除了拼接的那个维度，其他维度必须一样
```

---

# 20. stack 堆叠

## 优先级：⭐⭐⭐

```python
torch.stack([x1, x2], dim=0)
```

作用：

```text
新增一个维度，再堆起来
```

例子：

```text
x1: [B, F]
x2: [B, F]
```

```python
x = torch.stack([x1, x2], dim=0)
```

输出：

```text
[2, B, F]
```

区别：

```text
cat：不增加新维度
stack：增加新维度
```

---

# 21. BatchNorm / Dropout 通常不改变 Shape

## 优先级：⭐⭐⭐

## BatchNorm1d

```text
[B, F] → [B, F]
```

## BatchNorm2d

```text
[B, C, H, W] → [B, C, H, W]
```

## Dropout

```text
输入 shape = 输出 shape
```

例如：

```text
[B, 128] → [B, 128]
```

它们通常只改变数值，不改变 shape。

---

# 22. Softmax / Sigmoid 通常不改变 Shape

## 优先级：⭐⭐⭐

## Sigmoid

```text
[B, 1] → [B, 1]
```

## Softmax

```text
[B, num_classes] → [B, num_classes]
```

它们只把 logits 转换成概率形式，不改变 shape。

注意：

```text
训练时使用 CrossEntropyLoss 不需要手动 Softmax
训练时使用 BCEWithLogitsLoss 不需要手动 Sigmoid
```

---

# 23. Global Average Pooling

## 优先级：⭐⭐⭐

常用于 CNN 分类模型最后。

```python
nn.AdaptiveAvgPool2d((1, 1))
```

输入：

```text
[B, C, H, W]
```

输出：

```text
[B, C, 1, 1]
```

然后展平：

```text
[B, C, 1, 1] → [B, C]
```

再接分类层：

```text
[B, C] → [B, num_classes]
```

好处：

```text
可以适配不同大小的图片输入
```

---

# 24. 上采样 Upsample

## 优先级：⭐⭐

常用于图像分割、生成模型。

```python
nn.Upsample(scale_factor=2)
```

输入：

```text
[B, C, H, W]
```

输出：

```text
[B, C, 2H, 2W]
```

例如：

```text
[B, 64, 16, 16] → [B, 64, 32, 32]
```

---

# 25. 反卷积 ConvTranspose2d

## 优先级：⭐⭐

常用于分割、GAN、AutoEncoder。

```python
nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
```

输入：

```text
[B, 64, 16, 16]
```

输出：

```text
[B, 32, 32, 32]
```

解释：

```text
通道数: 64 → 32
高度宽度: 16 → 32
```

---

# 26. 图像分割任务 Shape

## 优先级：⭐⭐⭐

语义分割输入：

```text
image: [B, C, H, W]
```

模型输出：

```text
logits: [B, num_classes, H, W]
```

标签：

```text
mask: [B, H, W]
```

使用：

```python
nn.CrossEntropyLoss()
```

要求：

```text
y_pred: [B, num_classes, H, W]
y_true: [B, H, W]
```

注意：

```text
mask 不是 one-hot
mask 中每个像素是类别编号
```

---

# 27. 目标检测任务 Shape

## 优先级：⭐⭐

目标检测比分类复杂。

常见输入：

```text
image: [B, 3, H, W]
```

模型输出通常包括：

```text
类别预测
边界框预测
置信度预测
```

例如 YOLO 类模型可能输出：

```text
[B, num_anchors, grid_h, grid_w, 5 + num_classes]
```

其中：

```text
5 = x, y, w, h, confidence
```

例如：

```text
[B, 3, 13, 13, 85]
```

代表：

```text
B 个 batch
3 个 anchor
13x13 网格
每个框有 4 个坐标 + 1 个置信度 + 80 个类别
```

---

# 28. 常见报错与 Shape 原因

## 报错 1：mat1 and mat2 shapes cannot be multiplied

原因：

```text
Linear 层输入维度不对
```

常见问题：

```text
Flatten 后的维度和 Linear(in_features, out_features) 的 in_features 不一致
```

例子：

```text
x flatten 后是 [B, 2048]
但是 Linear 写成 nn.Linear(1024, 10)
```

应该改成：

```python
nn.Linear(2048, 10)
```

## 报错 2：Expected target size / Target size mismatch

原因：

```text
损失函数的预测值和标签 shape 不匹配
```

多分类：

```text
CrossEntropyLoss:
y_pred: [B, C]
y_true: [B]
```

二分类：

```text
BCEWithLogitsLoss:
y_pred: [B, 1]
y_true: [B, 1]
```

## 报错 3：Expected input batch_size to match target batch_size

原因：

```text
模型输出的 batch 数量和标签 batch 数量不一致
```

常见于 RNN 文本生成：

```text
y_pred: [B*T, V]
y_true: [B]
```

应该把标签也展平：

```text
y_true: [B, T] → [B*T]
```

## 报错 4：Expected object of scalar type Long

原因：

```text
CrossEntropyLoss 的标签类型错了
```

CrossEntropyLoss 要求：

```text
y_true.dtype = torch.long
```

不是：

```text
float
```

---

# 29. 最常见 Shape 流程图

## 29.1 表格分类任务

```text
原始数据
[B, F]
   ↓ Linear
[B, hidden]
   ↓ Linear
[B, num_classes]
   ↓ CrossEntropyLoss
target: [B]
```

## 29.2 图像分类任务

```text
图片输入
[B, 3, 32, 32]
   ↓ Conv2d
[B, 32, 32, 32]
   ↓ MaxPool2d
[B, 32, 16, 16]
   ↓ Conv2d
[B, 64, 16, 16]
   ↓ MaxPool2d
[B, 64, 8, 8]
   ↓ Flatten
[B, 64*8*8]
   ↓ Linear
[B, num_classes]
   ↓ CrossEntropyLoss
target: [B]
```

## 29.3 二分类任务

```text
输入
[B, F]
   ↓ Linear
[B, hidden]
   ↓ Linear
[B, 1]
   ↓ BCEWithLogitsLoss
target: [B, 1]
```

## 29.4 RNN 文本分类任务

```text
输入 token
[B, T]
   ↓ Embedding
[B, T, D]
   ↓ RNN
[B, T, hidden]
   ↓ 取最后时间步
[B, hidden]
   ↓ Linear
[B, num_classes]
   ↓ CrossEntropyLoss
target: [B]
```

## 29.5 RNN 文本生成任务

```text
输入 token
[B, T]
   ↓ Embedding
[B, T, D]
   ↓ RNN
[B, T, hidden]
   ↓ Linear
[B, T, V]
   ↓ reshape
[B*T, V]
   ↓ CrossEntropyLoss
target: [B*T]
```

---

# 30. 最重要的 Shape 检查口诀

## 优先级：⭐⭐⭐⭐⭐

```text
1. DataLoader 后，最前面一定多一个 B
2. CNN 输入必须是 [B, C, H, W]
3. Linear 只看最后一维
4. Flatten 时保留 B，合并后面所有维度
5. CrossEntropyLoss：预测 [B, C]，标签 [B]
6. BCEWithLogitsLoss：预测 [B, 1]，标签 [B, 1]
7. RNN 默认吃 [T, B, D]，batch_first=True 才吃 [B, T, D]
8. Embedding：整数编号 [B, T] → 词向量 [B, T, D]
9. reshape 不改变元素总数
10. squeeze 不要乱用，容易误删 batch 维度
```

---

# 31. 学习优先级建议

## 第一优先级：必须马上掌握

| 内容 | 原因 |
|---|---|
| DataLoader 加 batch | 所有训练代码都会遇到 |
| CNN 输入 `[B,C,H,W]` | 图像任务基础 |
| Linear 输入输出 | 全连接网络基础 |
| Flatten | CNN 接分类层必用 |
| CrossEntropyLoss shape | 多分类最常见 |
| BCEWithLogitsLoss shape | 二分类最常见 |

## 第二优先级：训练中经常遇到

| 内容 | 原因 |
|---|---|
| `reshape/view` | 手动调整 shape 必备 |
| `unsqueeze/squeeze` | 修正标签维度常用 |
| `permute/transpose` | 图像/RNN 经常用 |
| RNN shape | NLP 序列模型必备 |
| Embedding shape | NLP 入门必备 |

## 第三优先级：进阶任务掌握

| 内容 | 原因 |
|---|---|
| Transformer 多头维度变化 | 学大模型/Attention 必备 |
| 图像分割 shape | 分割任务需要 |
| 目标检测 shape | YOLO/Faster R-CNN 需要 |
| 上采样/反卷积 | 生成模型/分割模型需要 |

---

# 32. 一句话总结

深度学习里的 shape 改变，本质上只有四类：

```text
1. 增加维度：DataLoader、unsqueeze、stack
2. 删除维度：squeeze、pooling、flatten
3. 改变维度大小：Linear、Conv2d、Pooling、Embedding
4. 调换维度顺序：transpose、permute
```

最核心的检查方式：

```python
print(x.shape)
```

在每一层后面打印 shape，是排查深度学习代码错误最快的方法。
