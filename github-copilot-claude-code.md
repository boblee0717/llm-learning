# GitHub Copilot + Claude Code 配置指南

> 让 Claude Code 通过你已有的 GitHub Copilot 订阅运行，无需额外购买 Anthropic API。

## 原理

```
Claude Code CLI  ──>  ghc-api（本地代理）  ──>  GitHub Copilot API
                      localhost:8313            claude-sonnet-4 等模型
```

[sxwxs/ghc-api](https://github.com/sxwxs/ghc-api) 是一个 Python Flask 应用，在本地启动一个代理服务器，将 Claude Code 的请求翻译并转发给 GitHub Copilot API。它同时兼容 OpenAI 和 Anthropic 两种 API 格式，还自带 Web UI 管理界面。

## 前置条件

- Python 3（用于运行 ghc-api）
- Node.js / npm（用于安装 Claude Code CLI）
- 有效的 GitHub Copilot 订阅（individual / business / enterprise 均可）
- Claude Code CLI 已安装：
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```

## 安装与启动

### 第 1 步：安装 ghc-api

```bash
pip install ghc-api
```

### 第 2 步：生成配置文件

```bash
ghc-api --config
```

这会在 `~/.ghc-api/config.yaml`（Windows 为 `%APPDATA%/ghc-api/config.yaml`）生成默认配置。

### 第 3 步：启动代理

```bash
ghc-api
```

首次启动会触发 **GitHub Device Flow 认证**：终端会显示一个验证码和 URL，按提示在浏览器中完成 GitHub 登录授权即可。

认证成功后，代理默认运行在 `http://localhost:8313`。

## 配置 config.yaml

生成后的配置文件关键项：

```yaml
# 服务器设置
address: localhost
port: 8313
debug: false

# GitHub Copilot 账户类型（按你的订阅选择）
account_type: individual    # individual / business / enterprise

# 版本设置（构建请求头用，一般无需修改）
vscode_version: "1.93.0"
api_version: "2025-04-01"
copilot_version: "0.26.7"

# 模型名称映射
model_mappings:
  # 精确匹配（简写 -> 实际模型名）
  exact:
    opus: claude-opus-4.5
    sonnet: claude-sonnet-4.5
    haiku: claude-haiku-4.5
  # 前缀匹配（自动补全模型名）
  prefix:
    claude-sonnet-4-: claude-sonnet-4
    claude-opus-4.5-: claude-opus-4.5
```

根据你的 Copilot 订阅，修改 `account_type` 为对应类型即可，其余配置大多数场景无需改动。

## 配置 Claude Code 使用代理

编辑（或新建）`~/.claude/settings.json`，加入以下环境变量：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8313",
    "ANTHROPIC_AUTH_TOKEN": "not-needed"
  }
}
```

> 如果该文件已有其他配置，只需把 `env` 部分合并进去。

配置完成后，在任意项目目录中运行：

```bash
claude
```

Claude Code 的所有请求将自动通过 ghc-api 转发到 GitHub Copilot。

## 验证是否成功

1. **查看 Web UI**：浏览器打开 `http://localhost:8313`，能看到 ghc-api 的管理界面说明代理正常运行。
2. **观察终端日志**：在 Claude Code 中发送任意提问，ghc-api 的终端会打印出请求日志，确认流量走的是 Copilot 端点。
3. **检查模型**：在 Claude Code 对话中问「你是什么模型」，返回结果应能体现 Copilot 端的模型信息。

## 附加功能

### Web UI Agent 管理

ghc-api 的 `/agent` 页面提供基于浏览器的 Agent 交互界面，支持：

| Agent | 安装方式 |
|-------|---------|
| Claude Code | `npm install -g @agentclientprotocol/claude-agent-acp` |
| Codex | 从 GitHub Releases 下载 `codex-acp` |
| Copilot CLI | `npm install -g @github/copilot` |

### Config Sync（OneDrive）

ghc-api 支持通过 OneDrive 同步以下配置文件，方便多设备使用：

- `~/.claude/settings.json`（Claude Code）
- `~/.codex/config.toml`（Codex）
- `~/.ghc-api/config.yaml`（ghc-api 自身）

如不需要此功能，在 `config.yaml` 中设置 `disable_onedrive_access: true`。

### 请求日志

支持通过 API 导入导出请求历史记录，便于调试和审计。

## 日常使用流程

```bash
# 1. 启动代理（保持终端开着）
ghc-api

# 2. 在另一个终端，进入项目目录使用 Claude Code
cd your-project
claude

# 3. 用完后关闭代理（Ctrl+C 即可）
```

## 常见问题

### GitHub Device Flow 认证失败

- 检查你的 GitHub 账号是否有活跃的 Copilot 订阅
- 确认网络能访问 `github.com`
- 如果在公司网络下，检查是否需要配置 HTTP 代理

### 端口 8313 被占用

在 `~/.ghc-api/config.yaml` 中修改 `port` 为其他值，同时更新 `~/.claude/settings.json` 中的 `ANTHROPIC_BASE_URL`。

### 如何恢复直连 Anthropic

删除 `~/.claude/settings.json` 中的 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN`，Claude Code 会恢复使用 Anthropic 官方 API。

### Claude Code 报错 "connection refused"

确认 ghc-api 进程正在运行（`ghc-api` 命令启动后不要关闭终端）。

## 参考链接

- 仓库：[sxwxs/ghc-api](https://github.com/sxwxs/ghc-api)
- 官方文档：[Claude Code LLM Gateway](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)
- 官方文档：[Claude Code IDE 集成](https://docs.anthropic.com/en/docs/claude-code/ide-integrations)
