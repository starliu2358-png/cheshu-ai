# 知识资料维护

独立仓库没有 Gradio 管理页面。车型登记在 data/registry/vehicles.json，
文档登记在 data/registry/documents.json，正文在 data/knowledge。
维护时核对来源、车型、市场、版本和发布状态。

正文变更后重跑检索测试；向量模式执行 python -m scripts.build_vector_index。
vehicle_book/admin.py 和 dashboard.py 仅保留无 UI 的治理/评测数据辅助能力。
门禁说明见 [RELEASE_GATE.md](RELEASE_GATE.md)，机器指标不能代替语义核查。
