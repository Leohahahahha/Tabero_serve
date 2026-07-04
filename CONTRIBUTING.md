# 参与贡献 Tabero-VTLA

我们欢迎贡献、改进和修改。所有人都可以根据 [许可证](LICENSE) 使用 Tabero-VTLA。欢迎贡献者提交 bug 报告、功能请求和 Pull Request。由于我们是小团队，审核资源有限，无法保证批准所有 PR，但我们会尽最大努力。具体说明如下。

## 问题和功能请求

如果你想要讨论一些不直接涉及 bug 报告或功能请求的内容，欢迎使用 GitHub [Discussions](https://github.com/NathanWu7/Tabero-VTLA/discussions) 功能。这适合询问如何使用 Tabero-VTLA 的某些功能，或其他话题。

如果你发现了 bug 或其他问题，请先检查该问题是否已被报告（在 GitHub Issues 下使用搜索栏）。如果问题尚未报告，请在提交 GitHub Issue 时包含以下信息：

- 你的操作系统类型和版本，以及你使用的 Python 版本
- 允许我们复现 bug 的代码，包括所有依赖
- 异常的 Traceback
- 任何其他有助于我们排查的信息，例如截图

为了让我们能够解决任何问题，我们必须能够复现它。因此，如果你在修改 Tabero-VTLA 后遇到问题，请在不做任何其他修改的情况下复现该问题，并提供一个能让我们在 `main` 分支上快速复现问题的代码片段。

如果你想提交功能请求，请检查该功能请求是否已存在，并提供以下信息：

- 该功能的动机
- 你试图解决的问题或你的使用场景的描述
- 足够让我们理解请求性质的信息
- 关于你打算如何使用它的一些信息（这可能有助于我们理解动机！）

我们无法保证支持每个功能请求，但了解你感兴趣的使用场景对我们非常有帮助！

## 提交 Pull Request

如果你实现了对新机器人或环境的支持，或其他新功能，我们欢迎向 Tabero-VTLA 提交 Pull Request（PR）。我们鼓励你在开始编写 PR 之前先创建 [功能请求](https://github.com/NathanWu7/Tabero-VTLA/issues) 或在 [Discussions](https://github.com/NathanWu7/Tabero-VTLA/discussions) 上发帖，如果你想知道我们是否可能接受你的 PR。由于我们是小团队，维护和支持能力有限，我们可能不会接受所有 PR（例如，如果认为它会使代码更难维护，或审核 PR 超出我们的范围），因此提前联系我们是一个了解你的 PR 是否可能被合并到 Tabero-VTLA 的好方法。但即使没有被合并，你当然可以维护你自己的 fork，做任何你想要的修改。创建 PR 时，我们建议每个贡献都考虑以下几点：

- 确保你的 PR 有清晰的标题和描述
- 运行 `pre-commit`（首先使用 `pre-commit install` 安装），并运行 `ruff check .` 和 `ruff format .`
- 确保你的 PR 通过所有测试
