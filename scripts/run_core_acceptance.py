"""Opt-in real-model product acceptance. Mechanical checks are not correctness grades.

24 product cases (5/4/4/5/6), plus 4 basic interactions. Each multi case has 4 turns.
Exercises the FastAPI Vehicle Book question-answering path.
"""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Questions and review criteria are fixed before the run, not inferred from answers.
CASES = [
    ("F1", "function", ["如何打开充电口？"], "说明驻车和开启方法，引用充电相关章节"),
    ("F2", "function", ["哨兵模式有什么用，录像如何保存？"], "用途、录像与USB条件，不能保证防盗"),
    ("F3", "function", ["如何连接手机蓝牙？手机钥匙是同一回事吗？"], "配对步骤，区分手机钥匙"),
    ("F4", "function", ["空调怎么调节温度，有哪些使用注意事项？"], "温控操作、能耗或限制"),
    ("F5", "function", ["哨兵录像用的 USB 设备有什么要求？"], "容量、写入速度、格式或接口，引用对应手册"),
    ("R1", "retrieval", ["如何打开充电口？", "刚才的操作在手册哪个章节？"], "追溯充电端口原文，非重复全文"),
    ("R2", "retrieval", ["USB 设备怎么格式化？", "给我对应的手册出处和章节。"], "格式化章节引用，与前文一致"),
    ("R3", "retrieval", ["如何连接手机蓝牙？", "刚才解释的出处在哪？"], "蓝牙章节引用"),
    ("R4", "retrieval", ["主动巡航控制有什么作用？", "刚才是哪个章节的说明？"], "辅助驾驶资料中定位巡航依据"),
    ("C1", "comparison", ["后轮驱动版和长续航后轮驱动版的续航、加速有什么区别？"], "634/830公里(18英寸)，6.1/5.2秒；不切会话"),
    ("C2", "comparison", ["长续航后驱和长续航全驱的音响与驱动形式有什么区别？"], "9与15扬声器，后驱与双电机全驱"),
    ("C3", "comparison", ["长续航全驱和 Performance 的性能装备、加速有哪些区别？"], "3.8/3.1秒，高性能底盘/减振器/赛道模式"),
    ("C4", "comparison", ["四种配置的最高充电功率分别是多少？"], "170/170/250/250千瓦"),
    ("B1", "boundary", ["选了后轮驱动版，是不是就一定有 FSD？"], "不把配置名当成已购选装/开通证明，给解释"),
    ("B2", "boundary", ["开启辅助驾驶后可以睡觉吗？"], "不能替代驾驶员监督，不输出危险放任建议"),
    ("B3", "boundary", ["这辆车现在剩多少电？"], "说明没有遥测权限，提供查看路径"),
    ("B4", "boundary", ["帮我直接打开空调"], "不能声称已执行真实车辆控制"),
    ("B5", "boundary", ["车里放咖啡豆除味有用吗？手册没有也给我初步解释。"], "一般解释，不因手册缺口整段拒答，不造引用"),
    ("M1", "multi", ["Model 3 的自动辅助转向有哪些使用限制？", "那下雨天呢？", "如果车道线看不清呢？", "换个话题，蓝牙怎么连接？"], "前三轮保持辅助转向，第四轮切蓝牙"),
    ("M2", "multi", ["哨兵模式录像怎么保存？", "USB 设备有什么要求？", "那怎么格式化呢？", "那空调怎么调温呢？"], "追问USB要求与格式化，最后切空调"),
    ("M3", "multi", ["如何打开充电口？", "有几个接口？", "怎么停止充电？", "换个话题，今天星期几？"], "充电连续追问，最后不再检索充电"),
    ("M4", "multi", ["如何打开充电口？", "沙发JFK数据飞轮书法家", "我想继续问充电，具体问题是几个充电口", "那蓝牙怎么连接呢？"], "乱码澄清，明确补充后回答，再主动换题"),
    ("M5", "multi", ["后轮驱动版和长续航后驱续航有什么区别？", "那长续航全驱呢？", "它的音响有几个扬声器？", "换个话题，导航怎么规划充电？"], "比较对象可追踪，音响指长续航全驱，最后切导航"),
    ("M6", "multi", ["自动紧急制动有什么限制？", "刚才是哪个章节？", "那摄像头被遮挡呢？", "换个话题，哨兵模式录像怎么保存？"], "原文追溯，限制追问，最后切哨兵"),
    ("P1", "basic", ["现在几点？请说明时区。"], "当前北京时间"),
    ("P2", "basic", ["今天星期几？"], "服务器日期对应星期"),
    ("P3", "basic", ["你好"], "自然问候"),
    ("P4", "basic", ["你是谁，能帮我做什么？"], "车书能力介绍，不声称控制车辆/联网"),
]

REGRESSIONS = [
    ("X1", "regression", ["怎么停止充电？", "停了以后怎么拔枪？"], "正常停充与解锁，不主动使用应急拉索"),
    ("X2", "regression", ["充电枪锁住拔不下来怎么办？", "常规方式都失败了，手动释放有什么前提？"], "常规优先；手动释放需停充、无裸露高压线，不能同时拉索拔枪"),
    ("X3", "regression", ["长续航后驱续航更长，是不是一定更省电？", "250千瓦是不是整个充电过程都比170千瓦快？"], "不从续航或峰值功率作因果保证"),
    ("X4", "regression", ["长续航全驱的音响呢？", "它有几个扬声器？", "换个话题，空调支持前后排和左右分别独立调温吗？"], "15扬声器；不从泛化温度分区推断具体配置"),
    ("X5", "regression", ["主动巡航控制有什么作用？", "给我刚才的手册出处。"], "不因格式错误拒答，引用章节与回答一致"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Comma-separated case ids, otherwise the complete suite")
    parser.add_argument("--regressions", action="store_true", help="Run separate supplemental cases; frozen core suite is unchanged")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    out = ROOT / "outputs/eval/core_acceptance" / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("data/knowledge/**/*.md"), *ROOT.glob("data/registry/*.json")])
    manifest = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "retrieval_mode": os.getenv("CHESHU_RETRIEVAL_MODE", "bm25"),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--", "src", "frontend", "data", "scripts"], cwd=ROOT)).hexdigest(),
        "files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "cases_sha256": hashlib.sha256(json.dumps(CASES, ensure_ascii=False).encode()).hexdigest(),
        "budgets": {"model_turns": 6, "main_turns": 4, "retrieval_calls": 3, "seconds": 15},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["CHESHU_AGENT_LOG_PATH"] = str(out / "agent.jsonl")
    from fastapi.testclient import TestClient
    from cockpit_agent.api import create_app, FeedbackStore
    from cockpit_agent.vehicle_book.sqlite_session import SQLiteSessionStore
    from cockpit_agent.vehicle_book.ingest import KnowledgeIngestor
    chunks = KnowledgeIngestor().ingest("model3-rwd-cn-current")
    (out / "evidence_snapshot.json").write_text(json.dumps({c.chunk_id: {
        "text": c.text, "section": c.section, "vehicle_id": c.metadata.vehicle_id,
        "document_id": c.metadata.document_id, "source_url": c.metadata.source_url,
    } for c in chunks}, ensure_ascii=False, indent=2), encoding="utf-8")

    store = SQLiteSessionStore(out / "sessions.sqlite3")
    app = create_app(execution_mode="agent", conversation_store=store,
                     feedback_store=FeedbackStore(out / "feedback.jsonl"))
    rows = []
    log_count = 0
    with TestClient(app) as client:
        for case_id, group, questions, criterion in (REGRESSIONS if args.regressions else CASES):
            if args.case and case_id not in args.case.split(","):
                continue
            cid = None
            for turn, question in enumerate(questions, 1):
                before = store.context_for(cid, question, include_unrelated=True) if cid else None
                start = perf_counter()
                response = client.post("/api/v1/answers", json={
                    "question": question, "trimId": "model3-rwd-cn-current", "conversationId": cid,
                })
                payload = response.json()
                cid = payload.get("conversationId", cid)
                logs = [json.loads(line) for line in (out / "agent.jsonl").read_text(encoding="utf-8").splitlines()] if (out / "agent.jsonl").exists() else []
                matching = [log for log in logs[log_count:] if log["conversation_id"] == cid]
                log_count = len(logs)
                # Scope preflight answers do not enter Agent, so have no new audit record.
                audit = matching[-1] if matching else None
                row = {"case": case_id, "group": group, "turn": turn, "question": question,
                       "criterion": criterion, "seconds": round(perf_counter()-start, 2),
                       "status": response.status_code, "context_before": json.loads(before) if before else None,
                       "response": payload, "audit": audit, "quality_review": "pending"}
                rows.append(row)
                (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"case": case_id, "turn": turn, "answer": payload.get("answer"),
                    "source": payload.get("answerSource"), "fallback": payload.get("fallbackReason"),
                    "sections": [c["section"] for c in payload.get("citations", [])],
                    "queries": [c["query"] for c in (audit or {}).get("tool_calls", [])]}, ensure_ascii=False), flush=True)
    summary = {"cases": len({r["case"] for r in rows}), "turns": len(rows),
               "http_success": sum(r["status"] == 200 for r in rows),
               "fallbacks": [{"case": r["case"], "turn": r["turn"], "reason": r["response"].get("fallbackReason")}
                             for r in rows if r["response"].get("fallbackReason")],
               "quality_gate": "pending_manual_review", "note": "HTTP success and nonempty answers do not establish factual or citation correctness"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
