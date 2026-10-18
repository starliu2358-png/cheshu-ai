"""Opt-in live DeepSeek dialogue probe with isolated conversation storage."""
import json
import argparse
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUT = ROOT / "outputs" / "eval" / "conversation_style_live" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)
os.environ["CHESHU_AGENT_LOG_PATH"] = str(OUT / "agent.jsonl")

from fastapi.testclient import TestClient
from cockpit_agent.api import FeedbackStore, create_app
from cockpit_agent.vehicle_book.sqlite_session import SQLiteSessionStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--all-trims", action="store_true")
    args = parser.parse_args()
    app = create_app(
        execution_mode="agent",
        conversation_store=SQLiteSessionStore(OUT / "sessions.sqlite3"),
        feedback_store=FeedbackStore(OUT / "feedback.jsonl"),
    )
    cases = {
        "clock_followup": ["如何连接手机蓝牙？", "今天星期几？", "现在是几点钟？", "那东京现在几点？"],
        "unified_answers": ["你好", "17乘以23是多少？", "今天北京天气如何？", "车里放咖啡豆除味有用吗？", "沙发JFK数据飞轮书法家"],
        "trim_extended": ["我选的这款车驱动形式、18和19英寸轮毂的续航、音响扬声器数量，以及赛道模式分别是什么情况？"],
        "general_knowledge": [
            "如何连接手机蓝牙？", "今天星期几", "为什么天空是蓝色的？",
            "这辆车冬天充电慢一般可能是什么原因？如果手册没说明，给我一般性解释。",
        ],
        "trim_parameters": ["我选的这款车最高充电功率和百公里加速分别是多少？"],
        "manual_gap": ["车里放咖啡豆除味有用吗？手册没有的话也请给个初步解释。", "用完的咖啡渣呢？"],
        "charging_followup": [
            "如何打开充电口？",
            "有几个充电口？",
            "低压充电怎么操作？",
        ],
        "clarification_recovery": [
            "如何打开充电口？",
            "加萨力克金佛安抚是否能",
            "我想继续问充电，具体问题是几个充电口",
        ],
        "usb_followup": ["哨兵模式录像怎么保存？", "USB 设备有什么要求？"],
    }
    results = []
    cases = {f"{name}_{index + 1}": questions
             for name, questions in cases.items() if not args.case or name == args.case
             for index in range(args.repeat)}
    with TestClient(app) as client:
        trims = ["model3-rwd-cn-current"]
        if args.all_trims:
            trims += ["model3-lr-rwd-cn-current", "model3-lr-awd-cn-current", "model3-performance-awd-cn-current"]
        for name, questions, trim in [(n, q, t) for n, q in cases.items() for t in trims]:
            conversation = None
            for question in questions:
                start = perf_counter()
                response = client.post("/api/v1/answers", json={
                    "question": question, "trimId": trim,
                    "conversationId": conversation,
                })
                payload = response.json()
                conversation = payload.get("conversationId", conversation)
                row = {"case": name, "trim": trim, "question": question, "seconds": round(perf_counter()-start, 2),
                       "status": response.status_code, "response": payload}
                results.append(row)
                (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
