"""Regression checks for specific facts and their source/profile provenance."""
import pytest

from cockpit_agent.api import TESLA_CN_TRIM_IDS
from cockpit_agent.vehicle_book.retriever import VehicleBookRetriever


@pytest.mark.parametrize("vehicle_id", TESLA_CN_TRIM_IDS)
@pytest.mark.parametrize("question,source_suffix,facts", [
    ("几个充电口", "charging_interfaces_2024plus", ("两个接口", "交流", "直流")),
    ("哨兵模式 USB 设备要求", "usb_recording_2024plus", ("64 GB", "4 MB/s", "USB 2.0")),
    ("USB 格式化", "usb_recording_2024plus", ("exFAT", "TeslaCam", "备份")),
])
def test_retrieves_specific_official_facts_for_each_bound_trim(
    vehicle_id, question, source_suffix, facts
):
    results = VehicleBookRetriever().retrieve(vehicle_id, question, top_k=3)
    matched = [item.chunk for item in results
               if item.chunk.metadata.document_id.endswith(source_suffix)]
    assert matched, [item.chunk.chunk_id for item in results]
    assert all(item.chunk.metadata.vehicle_id == vehicle_id for item in results)
    assert all(item.metadata.market == "CN" and item.metadata.status == "published"
               for item in matched)
    text = "\n".join(item.text for item in matched)
    assert all(fact in text for fact in facts)
