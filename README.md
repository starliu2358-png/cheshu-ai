# 车书 AI · V1.0

React + FastAPI + OpenAI Agents SDK + DeepSeek 的单 Agent 车辆手册助手。
支持手册问答、引用查看、配置比较、连续追问、主动换题及 SQLite 本机会话持久化。

本项目不是 Tesla 官方产品，不读取实时车况、不控制真实车辆、不具备通用联网搜索。
回答可能出错，请核查重要信息；驾驶中请勿操作。本版本用于本机单用户体验，不应直接暴露公网。

## 能力与架构

- 单 Agent 调用证据型工具、改写查询；Python 限制最多 6 个模型轮次、3 次检索、15 秒总时限。
- 带引用回答经过模型证据复核；引用只能来自本轮证据。复核不保证完全正确。
- 默认 BM25；可选 BGE + FAISS + RRF 混合检索，Reranker 不默认启用。
- 结构化会话状态、简化记忆压缩；SQLite 保存对话、重命名、置顶和反馈。
- 四种 Model 3 配置只读展示；配置问题可在问题中注明版本。
- 麦克风打开语音菜单；点击开始输入才申请权限。自动播放默认关闭，本机浏览器记住偏好，保留手动播放。

展示四种配置不等于每次检索全部配置：API 仍保留会话绑定 Profile；初次会话默认列表首项后驱版，
后续新会话沿用当前绑定，旧会话不改变绑定。
Python 包名保留为 `cockpit_agent` 仅用于兼容；没有 Gradio、车辆控制或 MCP 演示链路。

## 安装与启动（PowerShell）

需要 Python 3.10+、Node.js 22.12+ 和自己的 DeepSeek API Key。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

填写 DEEPSEEK_API_KEY；DEEPSEEK_MODEL 以账户实际可用型号为准，随附验收使用 deepseek-flash。
不要提交或分享 .env。启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn cockpit_agent.api:app --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

打开 http://127.0.0.1:5173 ，API 文档为 http://127.0.0.1:8000/docs 。
前端代理 /api 到后端；`npm run build` 生成静态文件，但 FastAPI 不自动托管该目录。
仓库尚无公网认证、多用户权限、TLS 或生产部署方案。

### 可选混合检索

在根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
.\.venv\Scripts\python.exe -m scripts.build_vector_index
```

首次下载 BAAI/bge-small-zh-v1.5，需要网络和额外资源。
建索引后将 .env 中 CHESHU_RETRIEVAL_MODE 改为 hybrid，重启后端。
默认 BM25 无需下载向量模型；CHESHU_EXECUTION_MODE=rag 可回退固定 RAG。

## 测试与验收

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm test
npm run build
```

真实模型评测会产生 API 费用，在根目录执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_core_acceptance
.\.venv\Scripts\python.exe -m scripts.run_core_acceptance --regressions
```

随附三次真实模型验收（28 案例、各 50 轮）和 11 轮定向回归是拆仓前证据，
原始 manifest 指向原仓库提交和本地路径，不冒充新仓再次运行。
见 [验收报告](outputs/eval/core_acceptance/reliability_review_20261018.json)、
[版本说明](docs/RELEASE_NOTES_V1.0.md)、[已知问题](docs/KNOWN_ISSUES.md)。

## 数据、隐私与来源

会话保存在 outputs/state/cheshu.sqlite3；运行日志保存在 outputs/logs。
这些私有本地文件不随仓库分发。DeepSeek 会接收问题、上下文和检索片段，不要输入敏感信息。
自动播放偏好按浏览器保存，不跨设备同步。

资料登记在 data/registry/documents.json，正文在 data/knowledge。
Tesla 文档和商标归原权利人所有；整理资料不构成官方背书或对第三方材料的再授权。
OEM Alpha 是用于隔离测试的虚构车型数据，不是前端产品车型。

独立仓库使用全新 Git 历史。原历史保留在
[ai-cockpit-agent / feat/cheshu-ai-v1](https://github.com/starliu2358-png/ai-cockpit-agent/tree/feat/cheshu-ai-v1)，
基点 7c4264ae7b9ca70ed993ae19cd45c619779f806e。
V1.0 是车书产品版本，不是退回 BM25-only 技术阶段，也不代表全部严格门禁通过。
