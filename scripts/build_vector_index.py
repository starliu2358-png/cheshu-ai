"""Build deterministic BGE + FAISS indexes for registered vehicle books."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from cockpit_agent.vehicle_book.hybrid_retriever import VectorVehicleBookRetriever
from cockpit_agent.vehicle_book.registry import VehicleRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vehicle-id",
        action="append",
        help="vehicle ID to index; repeat for multiple vehicles (default: all released)",
    )
    parser.add_argument(
        "--include-pre-release",
        action="store_true",
        help="allow and include pre-release profiles",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = VehicleRegistry()
    vehicle_ids = args.vehicle_id or [
        profile.vehicle_id
        for profile in registry.list_profiles(
            include_pre_release=args.include_pre_release
        )
    ]
    retriever = VectorVehicleBookRetriever(vehicle_registry=registry)
    try:
        for vehicle_id in vehicle_ids:
            checksum = retriever.build_index(
                vehicle_id,
                include_pre_release=args.include_pre_release,
            )
            print(f"{vehicle_id}: {checksum}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
