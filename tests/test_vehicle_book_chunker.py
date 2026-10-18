from pathlib import Path

from cockpit_agent.vehicle_book.chunker import chunk_markdown
from cockpit_agent.vehicle_book.schemas import DocumentMetadata


def _document() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="aster_x1_owner_manual",
        vehicle_id="alpha_aster_x1_max_v1",
        title="ASTER X1 Owner Manual",
        content_type="owner_manual",
        document_version="1.0",
        source_path=Path("data/knowledge/owner_manual.md"),
        status="published",
        locale="zh-CN",
    )


def test_chunk_ids_are_deterministic_and_disambiguate_duplicate_headings() -> None:
    markdown = """# Overview

Introductory text.

## HVAC

First HVAC section.

## HVAC

Second HVAC section.
"""

    first = chunk_markdown(_document(), markdown)
    second = chunk_markdown(_document(), markdown)

    assert [chunk.chunk_id for chunk in first] == [
        "aster_x1_owner_manual:overview",
        "aster_x1_owner_manual:hvac",
        "aster_x1_owner_manual:hvac:2",
    ]
    assert [chunk.chunk_id for chunk in second] == [
        chunk.chunk_id for chunk in first
    ]


def test_headings_define_sections_and_empty_sections_are_ignored() -> None:
    markdown = """# Owner Manual

## Empty

## Seat Heating

Use levels 1 through 3.

### Warning

Keep the seat dry.
"""

    chunks = chunk_markdown(_document(), markdown)

    assert [chunk.section for chunk in chunks] == ["Seat Heating", "Warning"]
    assert chunks[0].text == "## Seat Heating\n\nUse levels 1 through 3."
    assert chunks[1].text == "### Warning\n\nKeep the seat dry."
    assert all(chunk.metadata == _document() for chunk in chunks)


def test_leading_front_matter_is_removed_without_parsing_yaml() -> None:
    markdown = """---
document_id: another_id
title: This metadata must not become searchable text
tags:
  - hvac
---

## Charging

Set the charge limit in the energy menu.
"""

    chunks = chunk_markdown(_document(), markdown)

    assert len(chunks) == 1
    assert chunks[0].section == "Charging"
    assert chunks[0].text == (
        "## Charging\n\nSet the charge limit in the energy menu."
    )
    assert "document_id" not in chunks[0].text
    assert "tags:" not in chunks[0].text
