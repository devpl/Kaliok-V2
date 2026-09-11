(() => {
  "use strict";

  const root = document.querySelector("[data-prototype-app]");
  if (!root) return;
  const payloadElement = document.getElementById("composer-data");
  const payload = payloadElement ? JSON.parse(payloadElement.textContent) : { pipelines: [] };
  const pipelines = Array.isArray(payload.pipelines) ? payload.pipelines : [];
  const state = { pipeline: null, revision: null, stepResults: {} };
  const apiBase = root.dataset.composerApiBase || "";

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  }
  function statusLabel(status) { return { active: "Active", draft: "Brouillon", retired: "Retirée" }[status] || status || "Inconnu"; }
  function badge(text, type = "neutral") { return `<span class="status-badge ${type}">${esc(text)}</span>`; }
  function route() { return (location.hash || "#/composer").replace(/^#\/?/, "").split("?")[0].split("/").filter(Boolean); }
  function pickInitialRevision(pipeline) {
    return pipeline?.revisions.find((item) => item.status === "draft") || pipeline?.revisions.find((item) => item.status === "active") || pipeline?.revisions[0] || null;
  }
  function selectPipeline(key) {
    const selected = pipelines.find((item) => item.key === key) || pipelines[0] || null;
    if (state.pipeline?.id !== selected?.id) { state.pipeline = selected; state.revision = pickInitialRevision(selected); }
    return selected;
  }
  function header(eyebrow, title, description, actions = "") {
    return `<div class="page-header"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><p>${esc(description)}</p></div>${actions ? `<div class="header-actions">${actions}</div>` : ""}</div>`;
  }
  function revisionSelect() {
    if (!state.pipeline) return "";
    return `<label class="revision-picker">Révision <select data-revision-select>${state.pipeline.revisions.map((revision) => `<option value="${esc(revision.id)}"${revision.id === state.revision?.id ? " selected" : ""}>v${revision.revision_number} · ${esc(statusLabel(revision.status))}</option>`).join("")}</select></label>`;
  }
  function componentName(assignment) {
    if (!assignment?.component) return "Composant indisponible";
    return `${assignment.component.display_name}${assignment.component_version?.version ? ` · ${assignment.component_version.version}` : ""}`;
  }
  function assignmentCard(assignment, selected = false) {
    const resource = assignment.resource_instance;
    const disabled = !assignment.enabled || !assignment.binding.enabled;
    return `<div class="assignment ${selected ? "selected" : "alternative"} ${disabled ? "disabled" : ""}"><div><span class="assignment-label">${selected ? "Binding sélectionné" : `Alternative · priorité ${assignment.priority}`}</span><strong>${esc(componentName(assignment))}</strong></div>${badge(disabled ? "Désactivé" : selected ? "Sélectionné" : "Non sélectionné", disabled ? "warn" : selected ? "good" : "neutral")}<span class="resource-line">Ressource : <strong>${resource ? esc(resource.display_name) : "Aucune ressource associée"}</strong>${resource?.capability_availability ? ` · ${esc(resource.capability_availability)}` : ""}</span>${assignment.issues.length ? `<ul class="issue-list">${assignment.issues.map((issue) => `<li>${esc(issue)}</li>`).join("")}</ul>` : ""}<details class="tech-details"><summary>Détails techniques</summary><div class="tech-content"><span><strong>Binding</strong><br>${esc(assignment.binding.key)}</span><span><strong>Configuration</strong><br>${assignment.configuration.keys.length ? esc(assignment.configuration.keys.join(", ")) : "Aucune"}</span></div></details></div>`;
  }
  function contractLabels(contracts) {
    return contracts.map((contract) => contract.artifact_type?.display_name || contract.artifact_type?.key || contract.port_key).join(", ") || "Non déclaré";
  }
  function nodeCard(node) {
    const selected = node.selected_binding;
    const stateText = !node.enabled ? "Node désactivé" : selected ? "Configuré" : "Non configuré";
    const canTest = Boolean(node.enabled && selected && selected.enabled && selected.binding.enabled);
    const result = state.stepResults[node.id];
    const resultHtml = result ? `<div class="missing-binding"><strong>${esc(result.status)}</strong><br>${esc(result.message || "")}</div>` : "";
    return `<article class="function-card ${selected ? "filled" : "empty"} ${!node.enabled ? "node-disabled" : ""}" data-node-id="${esc(node.id)}"><div class="function-top"><div><h3>${esc(node.display_name)}</h3><span class="function-kind">${esc(node.capability.display_name)} · ${node.requirement_mode === "optional" ? "Optionnel" : "Requis"}</span></div>${badge(stateText, !node.enabled || !selected ? "warn" : "good")}</div>${selected ? assignmentCard(selected, true) : `<div class="missing-binding">Aucun binding sélectionné pour cette fonction.</div>`}${node.alternatives.length ? `<div class="alternatives"><h4>Alternatives</h4>${node.alternatives.map((item) => assignmentCard(item)).join("")}</div>` : ""}${node.issues.length ? `<ul class="issue-list">${node.issues.map((issue) => `<li>${esc(issue)}</li>`).join("")}</ul>` : ""}<div class="card-actions"><button type="button" class="button" data-test-node="${esc(node.id)}" ${canTest ? "" : "disabled"} title="Exécute uniquement ce node">Tester cette étape</button></div>${resultHtml}<details class="tech-details"><summary>Contrats et identifiants</summary><div class="tech-content"><span><strong>Consomme</strong><br>${esc(contractLabels(node.capability.input_contracts))}</span><span><strong>Produit</strong><br>${esc(contractLabels(node.capability.output_contracts))}</span><span><strong>Node</strong><br>${esc(node.key)}</span><span><strong>Capability</strong><br>${esc(node.capability.key || "Inconnue")}</span></div></details></article>`;
  }
  function phaseView(node) {
    const phase = node.zone_key || node.capability.phase_key || "";
    if (["retrieval", "search", "indexing", "reranking"].includes(phase)) return "search";
    if (["generation", "response", "citation"].includes(phase)) return "response";
    return "preparation";
  }
  function emptyContent() {
    return `${header("Composer", "Aucun pipeline disponible", "PostgreSQL ne contient aucune PipelineDefinition à afficher.")}<div class="empty-state"><div><h3>Aucune donnée de composition</h3><p>La page reste en lecture seule et ne crée aucune donnée automatiquement.</p></div></div>`;
  }
  function renderOverview() {
    if (!state.pipeline || !state.revision) return emptyContent();
    const revision = state.revision;
    const configured = revision.nodes.filter((node) => node.selected_binding).length;
    const withoutResource = revision.nodes.filter((node) => node.selected_binding && !node.selected_binding.resource_instance).length;
    return `${header("Composer", state.pipeline.display_name, state.pipeline.description || "Composition RAG persistée.", revisionSelect())}<div class="overview-grid"><section class="panel"><div class="panel-title"><div><h2>Révision sélectionnée</h2><p>Projection exacte de la composition persistée.</p></div>${badge(statusLabel(revision.status), revision.status === "active" ? "good" : "warn")}</div><div class="overview-details"><div class="detail-row"><span class="detail-label">Pipeline</span><strong class="detail-value">${esc(state.pipeline.display_name)}</strong></div><div class="detail-row"><span class="detail-label">Révision</span><strong class="detail-value">v${revision.revision_number}</strong></div><div class="detail-row"><span class="detail-label">Modèle RAG</span><strong class="detail-value">${esc(revision.template?.display_name || "Aucun modèle associé")}</strong></div><div class="detail-row"><span class="detail-label">Révision du modèle</span><strong class="detail-value">${revision.template ? `v${revision.template.revision_number} · ${esc(statusLabel(revision.template.status))}` : "—"}</strong></div></div></section><section class="panel"><div class="panel-title"><div><h2>Couverture</h2><p>État descriptif, sans exécution runtime.</p></div></div><div class="status-list"><div class="status-item"><span class="status-name">Nodes</span>${badge(revision.nodes.length)}</div><div class="status-item"><span class="status-name">Bindings sélectionnés</span>${badge(`${configured} / ${revision.nodes.length}`, configured === revision.nodes.length ? "good" : "warn")}</div><div class="status-item"><span class="status-name">Edges déclarés</span>${badge(revision.edges.length)}</div><div class="status-item"><span class="status-name">Sans ressource</span>${badge(withoutResource, withoutResource ? "warn" : "good")}</div></div></section><section class="panel"><div class="panel-title"><div><h2>Fonctions</h2><p>Fonction d’abord, outil sélectionné ensuite.</p></div></div><div class="node-grid">${revision.nodes.map(nodeCard).join("")}</div></section><div class="mini-note"><span>◇</span><div><strong>Lecture seule</strong><br>Les actions de création, sélection, test et promotion seront raccordées dans des phases ultérieures.</div></div></div>`;
  }
  function renderZone(zone) {
    if (!state.pipeline || !state.revision) return emptyContent();
    const labels = { preparation: "Préparation", search: "Recherche", response: "Réponse" };
    const nodes = state.revision.nodes.filter((node) => phaseView(node) === zone);
    return `${header(`Composer · ${state.pipeline.display_name}`, labels[zone], "Vue de travail sur le même graphe RAG persisté.", revisionSelect())}<section class="lane-panel"><div class="lane-heading"><div><h2>${esc(labels[zone])}</h2><p>${nodes.length} fonction(s) issue(s) du modèle réel</p></div><span class="lane-tag">Lecture seule</span></div><div class="lane-body">${nodes.length ? nodes.map(nodeCard).join("") : `<div class="empty-state"><div><h3>Aucune fonction dans cette vue</h3><p>Aucun node persisté ne correspond à cette projection UX.</p></div></div>`}</div></section>`;
  }
  function renderGraph() {
    if (!state.pipeline || !state.revision) return emptyContent();
    const revision = state.revision;
    const byId = Object.fromEntries(revision.nodes.map((node) => [node.id, node]));
    return `${header(`Composition · ${state.pipeline.display_name}`, "Modèle RAG", "Nodes et edges proviennent de la RagTemplateRevision associée.", revisionSelect())}<div class="model-layout"><section class="model-canvas"><div class="model-meta"><div><h2>${esc(revision.template?.display_name || "Aucun modèle associé")}</h2><p>${revision.template ? `Révision ${revision.template.revision_number} · ${esc(statusLabel(revision.template.status))}` : "La PipelineRevision ne référence aucun template."}</p></div>${badge(`${revision.nodes.length} nodes · ${revision.edges.length} edges`)}</div><div class="graph-nodes">${revision.nodes.map((node) => `<div class="topology-function ${!node.enabled ? "disabled" : ""}"><div><strong>${esc(node.display_name)}</strong><span>${esc(node.capability.display_name)}</span></div>${badge(node.requirement_mode === "optional" ? "Optionnel" : "Requis")}</div>`).join("")}</div><div class="edge-list"><h3>Dataflow déclaré</h3>${revision.edges.length ? revision.edges.map((edge) => `<div class="edge-row ${!edge.enabled ? "disabled" : ""}"><strong>${esc(byId[edge.source_node_id]?.display_name || edge.source_node_id)}</strong><span>→</span><strong>${esc(byId[edge.target_node_id]?.display_name || edge.target_node_id)}</strong>${badge(edge.enabled ? edge.type : "Désactivé", edge.enabled ? "neutral" : "warn")}</div>`).join("") : `<p>Aucun edge déclaré pour cette révision de template.</p>`}</div></section><aside class="model-aside"><section class="panel"><div class="panel-title"><div><h3>Lecture du graphe</h3><p>Chaque edge est affiché indépendamment.</p></div></div><p>Cette vue accepte les branches, plusieurs prédécesseurs et plusieurs successeurs. Elle ne reconstruit pas la topologie depuis l’ordre des bindings.</p></section><div class="model-callout"><strong>Nouveau modèle</strong>La création reste désactivée tant que son workflow audité n’est pas implémenté.</div></aside></div>`;
  }
  function setupNavigation() {
    document.querySelector("[data-pipeline-navigation]").innerHTML = pipelines.map((pipeline) => `<a href="#/composer/pipeline/${encodeURIComponent(pipeline.key)}" data-pipeline-key="${esc(pipeline.key)}" class="nav-link pipeline-nav"><span class="pipeline-dot">${esc(pipeline.display_name.charAt(0).toUpperCase())}</span>${esc(pipeline.display_name)}<span class="nav-status">${pipeline.revisions.length} rév.</span></a>`).join("") || `<span class="nav-link disabled">Aucun pipeline</span>`;
    document.querySelector("[data-pipeline-menu]").innerHTML = pipelines.map((pipeline) => `<a href="#/composer/pipeline/${encodeURIComponent(pipeline.key)}">${esc(pipeline.display_name)} <span>${pipeline.revisions.length} rév.</span></a>`).join("");
  }
  function render() {
    const parts = route();
    if (parts[0] !== "composer") location.hash = "#/composer";
    if (parts[1] === "pipeline" && parts[2]) selectPipeline(decodeURIComponent(parts[2])); else if (!state.pipeline) selectPipeline();
    const section = ["model", "preparation", "search", "response"].includes(parts[1]) ? parts[1] : "composer";
    document.querySelector("[data-view]").innerHTML = section === "model" ? renderGraph() : ["preparation", "search", "response"].includes(section) ? renderZone(section) : renderOverview();
    document.querySelector("[data-breadcrumb]").innerHTML = `<a href="#/composer">Composer</a>${section !== "composer" ? `<span class="chevron">›</span><span class="current">${esc({ model: "Modèle RAG", preparation: "Préparation", search: "Recherche", response: "Réponse" }[section])}</span>` : ""}`;
    document.querySelectorAll("[data-nav]").forEach((item) => item.classList.toggle("active", item.dataset.nav === section));
    document.querySelectorAll("[data-pipeline-key]").forEach((item) => item.classList.toggle("active", item.dataset.pipelineKey === state.pipeline?.key));
    document.querySelector("[data-active-pipeline]").textContent = state.pipeline?.display_name || "Aucun";
    document.querySelector("[data-topbar-page]").textContent = section === "model" ? "Modèle RAG" : "Composer";
  }
  root.addEventListener("click", (event) => {
    if (event.target.closest("[data-action]")?.dataset.action === "toggle-pipeline-menu") { const menu = document.querySelector("[data-pipeline-menu]"); menu.hidden = !menu.hidden; }
    const button = event.target.closest("[data-test-node]");
    if (!button || !state.revision) return;
    const node = state.revision.nodes.find((item) => item.id === button.dataset.testNode);
    const raw = window.prompt("Entrées (une par ligne : type_artefact:uuid). Laissez vide pour vérifier les prérequis.", "");
    if (raw === null) return;
    const inputs = raw.split(/\n/).filter(Boolean).map((line) => { const [artifact_type_key, artifact_id] = line.trim().split(":"); return { artifact_type_key, artifact_id }; });
    button.disabled = true;
    fetch(`${apiBase}/rag/composer/execute-step`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pipeline_revision_id: state.revision.id, rag_template_node_id: node.id, inputs }) })
      .then((response) => response.json())
      .then((data) => { state.stepResults[node.id] = { status: data.status || "failed", message: data.error || data.unsupported_reason || (data.missing_inputs?.length ? `Entrée requise manquante : ${data.missing_inputs.map((item) => item.artifact_type_key).join(", ")}` : `Execution ${data.execution_id || ""}`) }; render(); })
      .catch(() => { state.stepResults[node.id] = { status: "failed", message: "Le service d'exécution est indisponible." }; render(); });
  });
  root.addEventListener("change", (event) => {
    if (!event.target.matches("[data-revision-select]")) return;
    state.revision = state.pipeline.revisions.find((item) => item.id === event.target.value) || state.revision;
    render();
  });
  window.addEventListener("hashchange", render);
  setupNavigation();
  render();
})();
