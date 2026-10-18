from __future__ import annotations

import re

from .schemas import DocumentMetadata, KnowledgeChunk

FRONT_MATTER_PATTERN = re.compile(
    r"\A\ufeff?---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)
HEADING_PATTERN = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _section_slug(section: str) -> str:
    """Return a readable, deterministic identifier fragment for a heading."""
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", section.lower()).strip("-")
    return slug or "section"


def chunk_markdown(
    document: DocumentMetadata, markdown: str
) -> tuple[KnowledgeChunk, ...]:
    """Split one Markdown document into deterministic heading-based chunks.

    Registry metadata is supplied separately and remains authoritative. A leading
    YAML-like front-matter block is removed without parsing it or requiring a YAML
    dependency.
    """
    content = FRONT_MATTER_PATTERN.sub("", markdown, count=1).strip()
    if not content:
        return ()

    headings = list(HEADING_PATTERN.finditer(content))
    if not headings:
        return (
            KnowledgeChunk(
                chunk_id=f"{document.document_id}:document",
                section=document.title,
                text=content,
                metadata=document,
            ),
        )

    chunks: list[KnowledgeChunk] = []
    slug_counts: dict[str, int] = {}

    for index, heading in enumerate(headings):
        body_start = heading.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        body = content[body_start:body_end].strip()
        if not body:
            continue

        section = heading.group(1).strip()
        base_slug = _section_slug(section)
        occurrence = slug_counts.get(base_slug, 0) + 1
        slug_counts[base_slug] = occurrence
        suffix = base_slug if occurrence == 1 else f"{base_slug}:{occurrence}"

        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document.document_id}:{suffix}",
                section=section,
                text=f"{heading.group(0).strip()}\n\n{body}",
                metadata=document,
            )
        )

    return tuple(chunks)
