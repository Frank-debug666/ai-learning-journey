# PyTorch 训练循环笔记

## 1. 核心概念

| 名称           | 含义                     | 直观理解                                 |
| -------------- | ------------------------ | ---------------------------------------- |
| **batch_size** | 每次送入模型训练的数据量 | 一次喂给模型多少条数据，例如 1024 张图片 |
| **iteration**  | 模型参数更新次数         | 每处理一个 batch 就算一次 iteration      |
| **epoch**      | 完整看完一次训练集       | 相当于把训练集完整跑一遍                 |

------

## 2. 三者关系

假设训练集有 50000 条数据，`batch_size = 1024`：

```
每个 epoch 的 iteration 数 = ceil(50000 / 1024) ≈ 49
如果训练 3 个 epoch，总 iteration = 3 × 49 = 147
```

- 每次 iteration 处理一个 batch，更新一次参数
- 每个 epoch 看完训练集所有数据
- batch_size 控制每个 iteration 的样本数量

------

## 3. 训练循环结构（示意）

```
for epoch in range(epochs):            # 外层循环：控制轮数
    for x_batch, y_batch in dataloader: # 内层循环：按 batch 遍历训练集
        # 前向传播
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1  # 每处理一个 batch 就算一次 iteration
```

------

## 4. Loss 和 Accuracy 的合理计算

```
batch_size = x_batch.shape[0]
total_loss += loss.item() * batch_size  # 累积总损失
total_true += (torch.argmax(y_pred, -1) == y_batch).sum().item()
total_num += batch_size

# 打印平均值
avg_loss = total_loss / total_num
accuracy = total_true / total_num
```

- 注意要乘以 `batch_size`，防止最后一个 batch 小于 batch_size 导致平均值偏差
- 打印或记录时，用 `total_loss / total_num` 计算平均 loss

------

## 5. 可视化理解

```
训练集（50000条）
        ↓ 按 batch_size=1024 切分
batch1  batch2  batch3  ...  batch49
  ↓       ↓       ↓            ↓
iter1   iter2   iter3        iter49
  \_______ 这 49 次 iteration 合起来 _______/
                  ↓
              1 个 epoch

如果训练 3 轮：
epoch1 + epoch2 + epoch3
总共 147 次 iteration
```

- 每个 batch = 1 次 iteration
- 每个 epoch = 训练集完整一轮
- batch_size = 每次喂入模型的样本数

------

## 6. 小贴士

- `iteration` 与 `optimizer.step()` 是一一对应的
- `batch_size` 越大，单次计算更稳定，但显存占用更高
- `epoch` 越多，模型越容易收敛，但可能过拟合
- 打印 loss / accuracy 时要考虑 batch_size，保证平均值准确