from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "src/kaliok/ui/core_ui/templates/core_ui/rag_prototype.html"
SCRIPT = ROOT / "src/kaliok/ui/core_ui/static/core_ui/rag_prototype.js"
STYLES = ROOT / "src/kaliok/ui/core_ui/static/core_ui/rag_prototype.css"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_comparative_shell_preserves_requested_navigation_without_overview_dashboard():
    template = _source(TEMPLATE)
    for label in (
        "KALIOK", "Composer", "Modèle RAG", "Préparation", "Recherche", "Réponse",
        "Ressources", "Outils", "Modèles IA", "Stockages", "Connexions",
        "Exécutions / Tests", "RAG Lab",
    ):
        assert label in template
    assert "data-pipeline-navigation" in template
    assert "data-workspace-meta" in template
    assert "Vue d’ensemble" not in template


def test_composer_renders_production_and_working_rag_from_distinct_real_revisions():
    script = _source(SCRIPT)
    assert 'item.status === "active"' in script
    assert 'item.status === "draft"' in script
    assert "RAG-P / Production" in script
    assert "RAG en construction" in script
    assert "matchedRows(production, candidate)" in script
    assert "node.capability.id === candidateNode.capability.id" in script
    assert "production?.nodes" in script
    assert "candidate?.nodes" in script
    assert "position + 1" not in script
    assert "synthetic" not in script.lower()


def test_working_card_exposes_real_compatible_tool_selection_directly():
    script = _source(SCRIPT)
    assert "capabilityCatalog(node)?.tools" in script
    assert "data-tool-select" in script
    assert "component_version_id: select.value" in script
    assert 'postMutation("nodes/select-tool/"' in script
    assert 'fetch("data/"' in script
    assert "Capabilities disponibles" in script
    assert "Utilisée ici" in script
    assert "capabilitiesForVersion" in script


def test_card_test_uses_existing_execution_endpoint_and_persisted_results():
    script = _source(SCRIPT)
    for capability in ("document_extraction", "normalization", "entity_discovery", "entity_resolution"):
        assert capability in script
    for label in ("Tester", "Retester", "Voir résultat", "Test indisponible"):
        assert label in script
    for output in ("ContentBlocks", "NormalizedContentUnits", "DiscoveredCandidates", "Entities"):
        assert output in script
    assert 'fetch("execute-step/"' in script
    assert "payload.lab_executions" in script
    assert "document_version_id === state.selectedDocumentId" in script
    assert "compatibleExecutions(node)" in script
    assert "window.prompt" not in script


def test_lower_workbench_is_adaptive_and_technical_details_are_collapsed():
    script = _source(SCRIPT)
    assert "Détails d’entrée" in script
    assert "Résultat détaillé" in script
    assert "Assemblage avancé" in script
    assert '<details class="technical-details"><summary>Détails techniques</summary>' in script
    assert "ExecutionStep" in script
    assert "ProcessingRun" in script
    assert "data-node-panel" in script
    assert "position: fixed" not in _source(STYLES)


def test_typography_and_comparison_layout_meet_readability_floor():
    styles = _source(STYLES)
    assert "font: 15px/1.5" in styles
    assert ".card-title h3" in styles
    assert "font-size: 18px" in styles
    assert ".comparison-board" in styles
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in styles
    assert "flex: 0 0 252px" in styles
    assert "@media (max-width: 980px)" in styles
    assert ":focus-visible" in styles


def test_graph_mutations_remain_real_and_connections_are_sober():
    script = _source(SCRIPT)
    assert "+ Ajouter une fonction" in script
    assert 'postMutation("nodes/"' in script
    assert 'postMutation("edges/"' in script
    assert "RagTemplateEdge persistés" in script
    assert "edge-layer" not in script
    assert "graph-edge" not in script
