# AI Learning Journey

一份持续更新的中文 AI 学习笔记与可运行代码示例，记录从机器学习、深度学习到大语言模型应用开发的学习过程。

本仓库面向正在入门 AI 的学习者。内容强调通俗解释、概念之间的联系，以及能够亲手运行的最小示例。

## 学习路线

推荐按照以下顺序学习：

1. [深度学习核心讲义：通俗版](docs/deep-learning/深度学习核心讲义_通俗版.md)
2. [神经网络搭建整体流程](docs/deep-learning/神经网络搭建整体流程.md)
3. [激活函数汇总](docs/deep-learning/激活函数汇总.md)
4. [损失函数汇总](docs/deep-learning/损失函数汇总_加入输入输出维度要求.md)
5. [优化器重点笔记](docs/deep-learning/优化器重点笔记_专业版.md)
6. [PyTorch 训练循环笔记](docs/deep-learning/PyTorch%20训练循环笔记.md)
7. [训练、评估与推理](docs/deep-learning/训练,评估,推理.md)
8. [RNN 的逻辑框架](docs/deep-learning/RNN的逻辑框架.md)
9. [Transformer 论文学习笔记](docs/llm/Transformer论文学习笔记.md)
10. [提示词书写宝典](docs/prompt-engineering/提示词书写宝典.md)

更完整的阶段目标请查看 [学习路线](docs/roadmap.md)。

## 代码示例

### OpenAI 兼容接口

示例使用阿里云百炼的 OpenAI 兼容接口，密钥只从环境变量读取。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写自己的 API Key，然后运行：

```powershell
python examples/openai-compatible-api/basic_chat.py
python examples/openai-compatible-api/stream_chat.py
```

## 仓库原则

- 只分享自己整理并有权公开的学习内容。
- 不提交 API Key、密码、个人信息、模型权重或完整数据集。
- 笔记会持续修订，错误欢迎通过 Issue 指出。
- 示例优先保持小而清晰，方便初学者理解。

## 内容声明

这些内容是个人学习过程中的理解与总结，可能存在疏漏，不应替代官方文档或专业建议。引用外部资料时会尽量注明来源。

## 许可证

- 示例代码采用 [MIT License](LICENSE)。
- 原创笔记采用 [CC BY 4.0](LICENSE-NOTES.md)。

欢迎学习、讨论与改进。
