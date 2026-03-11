from __future__ import annotations

import argparse

from linkedin_experiment.audit import run_dataset_audit
from linkedin_experiment.embeddings import train_group_models
from linkedin_experiment.pipeline import (
    build_labeling_outputs,
    build_robustness_outputs,
    build_scope_outputs,
)
from linkedin_experiment.weat import run_weat_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LinkedIn hiring-language experiment pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Profile the raw LinkedIn postings dataset.")
    audit_parser.add_argument("--nrows", type=int, default=None)

    scope_parser = subparsers.add_parser("scope", help="Build the digital-tech core corpus.")
    scope_parser.add_argument("--nrows", type=int, default=None)

    subparsers.add_parser("label", help="Apply seniority labels and export clean corpora.")

    train_parser = subparsers.add_parser("train", help="Train FastText models for each seniority group.")
    train_parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])

    weat_parser = subparsers.add_parser("weat", help="Run WEAT across the saved models.")
    weat_parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    weat_parser.add_argument("--permutations", type=int, default=1000)

    subparsers.add_parser("robustness", help="Write secondary robustness artifacts.")
    subparsers.add_parser("prep-all", help="Run scope build and labeling together.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        outputs = run_dataset_audit()
    elif args.command == "scope":
        outputs = build_scope_outputs(nrows=args.nrows)
    elif args.command == "label":
        outputs = build_labeling_outputs()
    elif args.command == "train":
        outputs = train_group_models(seeds=args.seeds)
    elif args.command == "weat":
        outputs = run_weat_suite(seeds=args.seeds, permutations=args.permutations)
    elif args.command == "robustness":
        outputs = build_robustness_outputs()
    elif args.command == "prep-all":
        build_scope_outputs()
        outputs = build_labeling_outputs()
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    print(outputs)


if __name__ == "__main__":
    main()
