from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEVICE = "cpu"
PASSAGE_COUNT = 5

QUESTION = (
    "Quelles sont les conditions de paiement et le prix total TTC "
    "annoncés pour le projet de véranda ?"
)

PASSAGES = [
    """
    Le devis relatif au projet de véranda présente les conditions financières
    convenues avec le client. Le règlement est échelonné de la manière
    suivante : 30 % du montant à la commande, 30 % lors du métrage définitif,
    30 % au démarrage de la pose, puis le solde à la fin de la pose après
    réception des travaux. Le tableau récapitulatif indique un prix hors taxes
    de 21 314,54 euros, une TVA à 10 % de 2 131,46 euros et un prix total TTC,
    livré et posé, de 23 446,00 euros. Ces montants comprennent la fabrication,
    la livraison et la pose des éléments décrits dans l'étude technique. Les
    travaux de maçonnerie préparatoires qui ne figurent pas dans le descriptif
    restent à la charge du client. La validité du devis, les modalités de
    réception et les réserves éventuelles sont rappelées dans les conditions
    générales. Le paiement de chaque échéance intervient sur présentation de
    la situation correspondante. En cas de modification demandée après le
    métrage, un avenant chiffré doit être accepté avant lancement en
    fabrication. Le document précise enfin que le prix TTC de 23 446,00 euros
    constitue le total de référence du projet tel qu'il est décrit.
    """,
    """
    L'étude technique décrit une véranda en aluminium de dimensions extérieures
    proches de 4 700 par 3 500 millimètres. Les profils sont réalisés dans un
    alliage 6060 T6 et reçoivent un traitement de surface répondant aux
    exigences QUALICOAT. La toiture est conçue en mono-pente avec une croupe,
    avec des panneaux isolants et des parties vitrées selon le plan de
    calepinage. Le document détaille les chéneaux, descentes d'eau, pièces de
    liaison, joints et dispositifs nécessaires à l'étanchéité de l'ensemble.
    Le vitrage de façade bénéficie d'une certification adaptée et les ouvrants
    comportent les accessoires prévus au descriptif. Les dimensions doivent
    être confirmées par le métreur avant toute mise en fabrication. Les plans
    et coupes restent indicatifs jusqu'à cette validation sur site. Plusieurs
    variantes de couleur, de remplissage et de ventilation peuvent être
    sélectionnées. Ce passage contient donc de nombreux éléments plausibles du
    même projet, mais il ne fournit ni l'échelonnement des paiements ni le prix
    TTC total demandé dans la question.
    """,
    """
    Le calendrier prévisionnel commence après la validation du dossier
    technique et le passage du métreur. La fabrication est annoncée dans un
    délai généralement compris entre dix et quatorze semaines, sous réserve de
    la disponibilité des matériaux et de l'achèvement des supports de
    maçonnerie. La date de pose est ensuite convenue avec le client en fonction
    de l'avancement du chantier. Les accès doivent être dégagés et permettre la
    manutention des éléments vitrés et des profilés de grande longueur. Une
    vérification des niveaux, aplombs et dimensions est réalisée avant
    intervention. Les éventuelles reprises du support doivent être terminées
    avant l'arrivée de l'équipe de pose. Les conditions météorologiques peuvent
    conduire à décaler certaines opérations. Le planning mentionne également
    les étapes de contrôle, de nettoyage et de réception. Ce passage concerne
    bien le même devis de véranda et pourrait sembler pertinent, mais son objet
    principal est le délai de fabrication et l'organisation de la pose, pas le
    montant total ni la répartition précise des échéances financières.
    """,
    """
    Les équipements de la véranda comprennent différents systèmes de
    fermeture, de ventilation et d'éclairage. Les coulissants peuvent recevoir
    une serrure à un ou trois points ainsi que des balais brosses assurant
    l'étanchéité à l'air. Des grilles de ventilation intégrées sont prévues sur
    certaines façades. Les spots incorporés à la toiture peuvent être commandés
    à distance et associés, en option, à une installation domotique. Les
    vitrages sont choisis en fonction des performances thermiques et du contrôle
    solaire recherchés. Le descriptif mentionne les joints, parcloses,
    habillages et accessoires assortis aux profilés aluminium. Chaque option
    doit être confirmée avant la commande définitive afin d'être intégrée aux
    plans de fabrication. Les schémas montrent l'implantation indicative des
    ouvrants et des éléments fixes. Bien que ce passage appartienne au projet
    et puisse contenir des termes proches de ceux du devis, il ne donne pas les
    conditions de paiement et ne présente aucun total TTC exploitable pour
    répondre précisément à la question posée.
    """,
    """
    Le rapport annuel consacré aux finances publiques décrit l'évolution des
    recettes, des dépenses et du déficit des administrations publiques. Il
    analyse la trajectoire de la dette, les hypothèses macroéconomiques et les
    mesures nécessaires au respect des engagements européens. Plusieurs
    scénarios sont comparés selon la croissance, l'inflation et le niveau des
    taux d'intérêt. Le texte présente également les risques pesant sur les
    collectivités territoriales et les organismes sociaux. Des tableaux
    synthétisent les prévisions budgétaires sur plusieurs exercices et les
    écarts observés avec les lois de programmation. Une partie est consacrée à
    la qualité de la dépense et aux méthodes d'évaluation des politiques
    publiques. Ce document ne concerne ni une véranda, ni un devis de travaux,
    ni un client particulier. Les montants évoqués sont des agrégats nationaux
    sans relation avec un prix livré et posé. Ce passage est volontairement non
    pertinent pour la question sur les conditions de paiement d'un projet de
    véranda.
    """,
]


@dataclass(frozen=True)
class WarmStatistics:
    average_batch_seconds: float
    average_passage_seconds: float


def calculate_warm_statistics(
    durations: list[float],
    passage_count: int,
) -> WarmStatistics:
    if len(durations) < 3 or passage_count <= 0:
        raise ValueError("Mesures insuffisantes pour la moyenne chaude.")
    warm_average = sum(durations[1:3]) / 2
    return WarmStatistics(
        average_batch_seconds=warm_average,
        average_passage_seconds=warm_average / passage_count,
    )


def rank_scores(scores: list[float]) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )


def _rss_mb(psutil_module: Any | None) -> float | None:
    if psutil_module is None:
        return None
    return psutil_module.Process().memory_info().rss / (1024 * 1024)


def _format_ram(value: float | None) -> str:
    return f"{value:.0f} MB" if value is not None else "non mesurée"


def _load_optional_psutil() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def _load_runtime() -> tuple[Any, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Dépendance absente : torch. Installez une version CPU de "
            "PyTorch compatible avec votre environnement."
        ) from error
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as error:
        raise RuntimeError(
            "Dépendance absente : FlagEmbedding. Installez-la explicitement "
            "dans l'environnement expérimental, par exemple avec "
            "'.\\.venv\\Scripts\\python.exe -m pip install FlagEmbedding'. "
            "Cette dépendance est volumineuse et n'est pas ajoutée au projet."
        ) from error
    return torch, FlagReranker


def run_benchmark() -> int:
    psutil_module = _load_optional_psutil()
    ram_before = _rss_mb(psutil_module)

    print(f"Modèle                 : {MODEL_NAME}")
    print(f"Device                 : {DEVICE}")

    try:
        torch, reranker_class = _load_runtime()
    except RuntimeError as error:
        print(f"Erreur de configuration : {error}", file=sys.stderr)
        return 2

    print(f"Torch threads          : {torch.get_num_threads()}")
    print(f"RAM avant chargement   : {_format_ram(ram_before)}")

    loading_started = time.perf_counter()
    try:
        reranker = reranker_class(
            MODEL_NAME,
            use_fp16=False,
            devices=[DEVICE],
        )
    except MemoryError as error:
        print(
            "Erreur de chargement : mémoire insuffisante pour charger le "
            f"modèle sur CPU ({error}).",
            file=sys.stderr,
        )
        return 3
    except OSError as error:
        print(
            "Erreur de chargement/téléchargement : impossible d'obtenir les "
            f"fichiers du modèle ({error}).",
            file=sys.stderr,
        )
        return 4
    except Exception as error:
        message = str(error)
        category = (
            "mémoire insuffisante probable"
            if "memory" in message.lower() or "allocate" in message.lower()
            else "erreur inattendue"
        )
        print(
            f"Erreur de chargement ({category}) : "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 5

    loading_duration = time.perf_counter() - loading_started
    ram_after = _rss_mb(psutil_module)
    model_ram = (
        max(0.0, ram_after - ram_before)
        if ram_after is not None and ram_before is not None
        else None
    )
    print(f"Chargement modèle      : {loading_duration:.2f} s")
    print(f"RAM après chargement   : {_format_ram(ram_after)}")
    print(f"RAM modèle approx.     : {_format_ram(model_ram)}")

    pairs = [[QUESTION, passage.strip()] for passage in PASSAGES]
    durations: list[float] = []
    scores: list[float] = []
    try:
        for batch_number in range(1, 4):
            started = time.perf_counter()
            raw_scores = reranker.compute_score(pairs, normalize=True)
            duration = time.perf_counter() - started
            durations.append(duration)
            scores = [float(score) for score in raw_scores]
            if len(scores) != PASSAGE_COUNT:
                raise RuntimeError(
                    "Nombre de scores inattendu : "
                    f"{len(scores)} au lieu de {PASSAGE_COUNT}."
                )
            print(
                f"Batch #{batch_number} - 5 passages  : {duration:.3f} s"
            )
    except MemoryError as error:
        print(
            f"Erreur pendant le scoring : mémoire insuffisante ({error}).",
            file=sys.stderr,
        )
        return 6
    except Exception as error:
        print(
            "Erreur pendant le scoring : "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 7

    statistics = calculate_warm_statistics(durations, PASSAGE_COUNT)
    print(
        "Moyenne batch chaud    : "
        f"{statistics.average_batch_seconds:.3f} s"
    )
    print(
        "Temps/passage chaud    : "
        f"{statistics.average_passage_seconds:.3f} s"
    )
    print("\nScores :")
    for page, score in enumerate(scores, start=1):
        print(f"page {page} : {score:.6f}")
    ranking = [index + 1 for index in rank_scores(scores)]
    print(f"\nClassement :\n{ranking}")
    print(f"\nRAM finale             : {_format_ram(_rss_mb(psutil_module))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
