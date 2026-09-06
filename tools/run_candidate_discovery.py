from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from sqlmodel import Session

from kaliok.discovery.experiment import run_candidate_discovery_experiment
from kaliok.storage.database import create_database_engine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exécute Candidate Discovery sur une version existante."
    )
    document = parser.add_mutually_exclusive_group(required=True)
    document.add_argument("--document-version-id", type=UUID)
    document.add_argument("--filename", help="Nom complet ou recherche partielle.")
    parser.add_argument("--normalization-run-id", type=UUID)
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=Path("config/discovery/development_dictionary.json"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--commit",
        action="store_true",
        help="Persiste volontairement le run et ses occurrences.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Annule l'expérience après affichage (comportement par défaut).",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = _parser().parse_args()
    engine = create_database_engine()
    try:
        with Session(engine) as session:
            try:
                summary = run_candidate_discovery_experiment(
                    session,
                    dictionary_path=arguments.dictionary,
                    document_version_id=arguments.document_version_id,
                    filename=arguments.filename,
                    normalization_run_id=arguments.normalization_run_id,
                    commit=arguments.commit,
                )
            except Exception as error:
                session.rollback()
                print(f"Erreur : {error}", file=sys.stderr)
                return 1
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
