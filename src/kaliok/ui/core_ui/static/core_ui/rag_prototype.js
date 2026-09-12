(() => {
  "use strict";
  const root = document.querySelector("[data-prototype-app]");
  if (!root) return;
  const source = document.getElementById("composer-data");
  let payload = source ? JSON.parse(source.textContent) : { pipelines: [], capabilities: [] };
  let pipelines = Array.isArray(payload.pipelines) ? payload.pipelines : [];
  const persistedDocument = localStorage.getItem("kaliok-rag-lab-document");
  const state = { pipeline: null, revision: null, selectedNodeId: null, selectedDocumentId: persistedDocument || payload.documents?.[0]?.id || null, stepResults: {}, showAddFunction: false, mutationBusy: false, mutationError: "" };
  const csrfToken = document.querySelector("[name=csrfmiddlewaretoken]")?.value || "";
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  const labels = { active: "Production", draft: "Brouillon", retired: "Retirée", running: "En cours", completed: "Réussi", failed: "Échec", refused: "Refusé" };
  const statusLabel = (value) => labels[value] || value || "Inconnu";
  const badge = (text, kind = "neutral") => `<span class="status-badge ${kind}"><i aria-hidden="true"></i>${esc(text)}</span>`;
  const route = () => (location.hash || "#/composer").replace(/^#\/?/, "").split("?")[0].split("/").filter(Boolean);
  const executableCapabilities = new Set(["document_extraction", "normalization", "entity_discovery", "entity_resolution"]);
  const outputLabels = { document_extraction: "ContentBlocks", normalization: "NormalizedContentUnits", entity_discovery: "DiscoveredCandidates", entity_resolution: "Entities" };

  function productionRevision() { return state.pipeline?.revisions.find((item) => item.status === "active") || null; }
  function pickRevision(pipeline) { return pipeline?.revisions.find((item) => item.status === "draft") || pipeline?.revisions.find((item) => item.status === "active") || pipeline?.revisions[0] || null; }
  function selectPipeline(key) {
    const selected = pipelines.find((item) => item.key === key) || pipelines[0] || null;
    if (state.pipeline?.id !== selected?.id) { state.pipeline = selected; state.revision = pickRevision(selected); state.selectedNodeId = null; restoreExecutions(); }
  }
  function capabilityCatalog(node) { return (payload.capabilities || []).find((item) => item.id === node?.capability?.id); }
  function capabilitiesForVersion(componentVersionId) { return componentVersionId ? (payload.capabilities || []).filter((capability) => (capability.tools || []).some((tool) => tool.component_version_id === componentVersionId)) : []; }
  function lastExecution(node, revisionId = state.revision?.id) {
    return (payload.lab_executions || []).find((item) => item.pipeline_revision_id === revisionId && item.rag_template_node_id === node.id && (!state.selectedDocumentId || item.document_version_id === state.selectedDocumentId)) || null;
  }
  function restoreExecutions() {
    state.stepResults = {};
    if (!state.revision) return;
    for (const node of state.revision.nodes || []) {
      const item = lastExecution(node);
      if (item) state.stepResults[node.id] = { ...item, inputs: (item.input_artifacts || []).map((artifact) => artifact.id), outputs: (item.output_artifacts || []).map((artifact) => artifact.id), processing_runs: item.processing_runs || [] };
    }
  }
  function compatibleExecutions(node) {
    const accepted = new Set((node.capability.input_contracts || []).map((item) => item.artifact_type?.key));
    return (payload.lab_executions || []).filter((item) => item.pipeline_revision_id === state.revision?.id && item.document_version_id === state.selectedDocumentId && item.status === "completed").flatMap((item) => (item.output_artifacts || []).filter((artifact) => accepted.has(artifact.artifact_type_key)).map((artifact) => ({ ...artifact, sourceNodeId: item.rag_template_node_id, executionId: item.execution_id })));
  }
  function phaseView(node) {
    const phase = node.zone_key || node.capability.phase_key || "";
    if (["retrieval", "search", "indexing", "reranking"].includes(phase)) return "search";
    if (["generation", "response", "citation"].includes(phase)) return "response";
    return "preparation";
  }
  function componentName(assignment) { return assignment?.component?.display_name || "Aucun outil sélectionné"; }
  function toolVersion(assignment) { return assignment?.component_version?.version || "Version non renseignée"; }
  function toolName(assignment) { return assignment ? `${componentName(assignment)} ${toolVersion(assignment)}` : "Non configuré"; }
  function isConfigured(node) { const assignment = node.selected_binding; return Boolean(node.enabled && assignment && assignment.enabled && assignment.binding.enabled); }
  function metricFor(node, result) {
    if (!result) return null;
    const metrics = result.metrics || {};
    const value = metrics.output_count ?? (result.output_artifacts || []).length ?? (result.outputs || []).length;
    return { value, label: outputLabels[node.capability.key] || "sorties" };
  }
  function duration(result) { return !result || result.duration_ms == null ? "—" : result.duration_ms >= 1000 ? `${(result.duration_ms / 1000).toFixed(2).replace(".", ",")} s` : `${result.duration_ms} ms`; }

  function productionCard(node) {
    if (!node) return `<div class="comparison-placeholder"><span>Pas de fonction équivalente en Production</span></div>`;
    const result = lastExecution(node, productionRevision()?.id);
    const metric = metricFor(node, result);
    const assignment = node.selected_binding;
    const bindingStatus = assignment ? "Déclaré dans Composer" : "Binding Composer : non encore déclaré";
    const activeTool = assignment ? toolName(assignment) : "Outil runtime réel non déterminé";
    return `<article class="function-card production-card" data-production-node="${esc(node.id)}"><div class="card-title"><div><span class="card-kicker">${esc(node.capability.display_name)}</span><h3>${esc(node.display_name)}</h3></div>${badge("Production", "success")}</div><div class="production-tool"><span>Outil réel utilisé</span><strong>${esc(activeTool)}</strong><span>Capability utilisée : ${esc(node.capability.display_name)}</span><small>${esc(bindingStatus)}</small></div>${assignment ? capabilityDisclosure(node, assignment.component_version?.id) : ""}${result ? `<div class="compact-result"><span>Dernière information</span><strong>${esc(duration(result))}${metric ? ` · ${esc(metric.value)} ${esc(metric.label)}` : ""}</strong></div>` : ""}</article>`;
  }
  function capabilityDisclosure(node, componentVersionId) {
    const available = capabilitiesForVersion(componentVersionId);
    return available.length ? `<div class="capability-disclosure"><span>Capabilities disponibles : ${available.map((item) => esc(item.display_name)).join(" · ")}</span><strong>Utilisée ici : ${esc(node.capability.display_name)}</strong></div>` : "";
  }
  function inlineToolSelector(node) {
    const tools = capabilityCatalog(node)?.tools || [];
    const selectedId = node.selected_binding?.component_version?.id;
    if (state.revision?.status !== "draft") return `<div class="readonly-tool"><span>Outil</span><strong>${esc(toolName(node.selected_binding))}</strong></div>`;
    const choices = tools.map((tool) => `${esc(tool.display_name)}${tool.version ? ` v${esc(tool.version)}` : ""}${tool.document_capabilities?.length ? ` — structure, ${tool.document_capabilities.map(esc).join(", ")}` : ""}`).join(" · ");
    return `<label class="inline-tool"><span>Outil</span><select data-tool-select data-node-id="${esc(node.id)}" ${state.mutationBusy || !tools.length ? "disabled" : ""}>${!tools.length ? `<option>Aucun outil compatible</option>` : `${selectedId ? "" : `<option value="" selected disabled>Sélectionner un outil compatible</option>`}${tools.map((tool) => `<option value="${esc(tool.component_version_id)}"${tool.component_version_id === selectedId ? " selected" : ""}${tool.status !== "available" ? " disabled" : ""}>${esc(tool.display_name)}${tool.version ? ` v${esc(tool.version)}` : ""}</option>`).join("")}`}</select>${choices ? `<small>Outils compatibles : ${choices}</small>` : ""}</label>`;
  }
  function testUnavailable(node) {
    if (!isConfigured(node)) return !node.enabled ? "Fonction désactivée" : "Sélectionnez d’abord un outil compatible";
    if (executableCapabilities.has(node.capability.key) && node.lab_execution?.executable) return "";
    return node.lab_execution?.reason || "Cette fonction ne possède pas encore de test isolé.";
  }
  function inlineTest(node) {
    const result = state.stepResults[node.id] || lastExecution(node);
    const reason = testUnavailable(node);
    const running = result?.status === "running";
    const metric = metricFor(node, result);
    if (reason) return `<div class="inline-test unavailable"><strong>Test indisponible</strong><span>${esc(reason)}</span></div>`;
    return `<div class="inline-test" aria-live="polite">${result ? `<div class="test-facts">${badge(statusLabel(result.status), result.status === "completed" ? "success" : result.status === "running" ? "running" : "warning")}<strong>${esc(duration(result))}</strong>${metric ? `<strong>${esc(metric.value)} <span>${esc(metric.label)}</span></strong>` : ""}</div>` : `<span class="not-tested">Pas encore testé avec ce document</span>`}<div class="card-actions"><button type="button" class="test-button" data-test-node="${esc(node.id)}" ${running ? "disabled" : ""}>${running ? "Test en cours…" : result ? "Retester" : "Tester"}</button>${result ? `<button type="button" class="text-action" data-select-node="${esc(node.id)}">Voir résultat</button>` : ""}</div></div>`;
  }
  function candidateCard(node) {
    if (!node) return `<div class="comparison-placeholder candidate"><span>Fonction présente uniquement en Production</span></div>`;
    const selectedId = node.selected_binding?.component_version?.id;
    return `<article class="function-card candidate-card ${state.selectedNodeId === node.id ? "selected" : ""}" data-candidate-node="${esc(node.id)}"><div class="card-title"><div><span class="card-kicker">${esc(node.capability.display_name)}</span><h3>${esc(node.display_name)}</h3></div>${badge(isConfigured(node) ? "Configuré" : "À configurer", isConfigured(node) ? "success" : "warning")}</div>${inlineToolSelector(node)}${capabilityDisclosure(node, selectedId)}${inlineTest(node)}</article>`;
  }
  function matchedRows(production, candidate) {
    const left = [...(production?.nodes || [])]; const right = [...(candidate?.nodes || [])]; const usedLeft = new Set(); const rows = [];
    for (const candidateNode of right) { const matchIndex = left.findIndex((node, index) => !usedLeft.has(index) && node.capability.id === candidateNode.capability.id); if (matchIndex >= 0) usedLeft.add(matchIndex); rows.push({ production: matchIndex >= 0 ? left[matchIndex] : null, candidate: candidateNode }); }
    left.forEach((node, index) => { if (!usedLeft.has(index)) rows.push({ production: node, candidate: null }); });
    return rows;
  }
  function filterRevision(revision, section) { return !revision || !["preparation", "search", "response"].includes(section) ? revision : { ...revision, nodes: revision.nodes.filter((node) => phaseView(node) === section) }; }

  function compactHeader() {
    const production = productionRevision(); const revision = state.revision; const documents = payload.documents || [];
    return `<section class="composer-header" aria-label="Contexte du Composer"><div class="header-field"><span>Pipeline</span><strong>${esc(state.pipeline?.display_name || "—")}</strong></div><label class="header-field"><span>Révision de travail</span><select data-revision-select>${(state.pipeline?.revisions || []).filter((item) => item.status === "draft").map((item) => `<option value="${esc(item.id)}"${item.id === revision?.id ? " selected" : ""}>v${item.revision_number}</option>`).join("") || `<option value="${esc(revision?.id || "")}">v${esc(revision?.revision_number || "—")}</option>`}</select></label><div class="header-field"><span>Statut</span>${badge(revision ? statusLabel(revision.status) : "Indisponible", revision?.status === "draft" ? "neutral" : "warning")}</div><div class="header-field mode"><span>Mode</span><strong>${production ? "Production / RAG de travail" : "RAG de travail"}</strong></div><label class="header-field document"><span>DocumentVersion de test</span><select data-document-select ${revision?.status !== "draft" ? "disabled" : ""}>${documents.map((item) => `<option value="${esc(item.id)}"${item.id === state.selectedDocumentId ? " selected" : ""}>${esc(item.title || item.filename)} · v${item.version_number}</option>`).join("")}</select></label></section>${state.mutationError ? `<div class="mutation-error" role="alert">${esc(state.mutationError)}</div>` : ""}`;
  }
  function mutationToolbar() {
    const capabilities = Array.isArray(payload.capabilities) ? payload.capabilities : []; const draft = state.revision?.status === "draft";
    return `<div class="mutation-toolbar">${draft ? `<button type="button" class="secondary-action" data-toggle-add-function aria-expanded="${state.showAddFunction}">+ Ajouter une fonction</button>` : ""}<button type="button" class="secondary-action" data-fork-revision ${state.mutationBusy ? "disabled" : ""}>Créer depuis Production</button><button type="button" class="secondary-action" data-empty-revision ${state.mutationBusy ? "disabled" : ""}>Créer depuis zéro</button>${draft && state.showAddFunction ? `<form class="add-function-form" data-add-function-form><label><span>Fonction réelle</span><select name="capability_id" required>${capabilities.map((item) => `<option value="${esc(item.id)}">${esc(item.display_name)}</option>`).join("")}</select></label><button type="submit" class="primary-action" ${state.mutationBusy || !capabilities.length ? "disabled" : ""}>Ajouter</button></form>` : ""}</div>`;
  }
  function connectionSummary(revision) { return revision ? `<div class="connection-summary"><strong>${revision.edges.length} connexion${revision.edges.length === 1 ? "" : "s"}</strong><span>RagTemplateEdge persistés · ouvrir une fonction pour l’assemblage avancé</span></div>` : ""; }
  function technicalPanel(node) {
    if (!node) return "";
    const assignment = node.selected_binding; const result = state.stepResults[node.id] || lastExecution(node); const candidates = compatibleExecutions(node);
    const inputNames = (node.capability.input_contracts || []).map((item) => item.artifact_type?.display_name || item.artifact_type?.key).filter(Boolean);
    const outputNames = (node.capability.output_contracts || []).map((item) => item.artifact_type?.display_name || item.artifact_type?.key).filter(Boolean);
    const sources = state.revision.nodes.filter((sourceNode) => sourceNode.id !== node.id && (sourceNode.capability.output_contracts || []).some((contract) => (node.capability.input_contracts || []).some((input) => input.artifact_type?.key === contract.artifact_type?.key)));
    return `<section class="function-workbench" data-node-panel><div class="workbench-heading"><div><span>Détails de la fonction</span><h2>${esc(node.display_name)}</h2></div><button type="button" data-close-panel aria-label="Fermer">×</button></div><div class="adaptive-details"><div><span>Entrée</span><strong>${node.capability.key === "document_extraction" ? "DocumentVersion sélectionnée" : esc(inputNames.join(", ") || "Aucune")}</strong></div><div><span>Sortie</span><strong>${esc(outputNames.join(", ") || "Aucune")}</strong></div>${result ? `<div><span>Dernier résultat</span><strong>${esc(statusLabel(result.status))} · ${esc(duration(result))}</strong></div>` : ""}</div>${node.capability.key !== "document_extraction" ? `<details class="advanced-inputs"><summary>Détails d’entrée</summary>${candidates.length ? `<div class="artifact-picker">${candidates.map((item) => `<label><input type="checkbox" data-step-artifact value="${esc(item.artifact_type_key)}:${esc(item.id)}" checked> ${esc(item.artifact_type_key)} · ${esc(item.id)}</label>`).join("")}</div>` : `<p>Aucun artefact compatible produit pour ce document.</p>`}<textarea data-step-input rows="3" placeholder="artifact_type:uuid"></textarea></details>` : ""}${result ? `<details class="result-details" open><summary>Résultat détaillé</summary><dl>${Object.entries(result.metrics || {}).map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(typeof value === "object" ? JSON.stringify(value) : value)}</dd></div>`).join("")}</dl></details>` : ""}${state.revision.status === "draft" ? `<details class="assembly-details"><summary>Assemblage avancé</summary>${sources.length ? `<form data-edge-form data-target-node="${esc(node.id)}"><label><span>Fonction source compatible</span><select name="source_node_id">${sources.map((item) => `<option value="${esc(item.id)}">${esc(item.display_name)}</option>`).join("")}</select></label><button type="submit" class="secondary-action">Créer la connexion réelle</button></form>` : `<p>Aucune fonction source compatible dans le graphe.</p>`}</details>` : ""}<details class="technical-details"><summary>Détails techniques</summary><dl><div><dt>Node</dt><dd>${esc(node.id)}</dd></div><div><dt>Capability</dt><dd>${esc(node.capability.id)}</dd></div>${assignment ? `<div><dt>Binding</dt><dd>${esc(assignment.binding.id)}</dd></div><div><dt>ComponentVersion</dt><dd>${esc(assignment.component_version.id)}</dd></div>` : ""}${result ? `<div><dt>Execution</dt><dd>${esc(result.execution_id || "—")}</dd></div><div><dt>ExecutionStep</dt><dd>${esc(result.execution_step_id || "—")}</dd></div><div><dt>ProcessingRun</dt><dd>${esc((result.processing_runs || []).join(", ") || "—")}</dd></div>` : ""}</dl></details></section>`;
  }
  function renderComposer(section) {
    if (!state.pipeline || !state.revision) return `${compactHeader()}<div class="empty-state">Aucun pipeline disponible.</div>`;
    const production = filterRevision(productionRevision(), section); const candidate = filterRevision(state.revision, section); const rows = matchedRows(production, candidate); const selectedNode = state.revision.nodes.find((node) => node.id === state.selectedNodeId) || null;
    return `${compactHeader()}${mutationToolbar()}<section class="comparison-board"><div class="column-heading production"><span>Référence en lecture seule</span><h1>RAG-P / Production</h1><p>${production ? `v${production.revision_number} · ${production.nodes.length} fonction${production.nodes.length === 1 ? "" : "s"}` : "Aucune révision active"}</p></div><div class="column-heading candidate"><span>Construction persistée</span><h1>RAG en construction</h1><p>v${candidate.revision_number} · ${candidate.nodes.length} fonction${candidate.nodes.length === 1 ? "" : "s"}</p></div><div class="comparison-rows">${rows.map((row) => `<div class="comparison-row">${productionCard(row.production)}${candidateCard(row.candidate)}</div>`).join("") || `<div class="comparison-empty">Aucune fonction dans cette vue. Utilisez « Ajouter une fonction » pour construire le RAG.</div>`}</div><div class="column-connection">${connectionSummary(production)}</div><div class="column-connection">${connectionSummary(candidate)}</div></section>${technicalPanel(selectedNode)}`;
  }
  function setupNavigation() {
    document.querySelector("[data-pipeline-navigation]").innerHTML = pipelines.map((pipeline) => `<a href="#/composer/pipeline/${encodeURIComponent(pipeline.key)}" data-pipeline-key="${esc(pipeline.key)}" class="nav-link pipeline-nav"><span class="pipeline-dot">${esc(pipeline.display_name.charAt(0).toUpperCase())}</span><span>${esc(pipeline.display_name)}</span><small>${pipeline.revisions.length} rév.</small></a>`).join("") || `<span class="nav-link disabled">Aucun pipeline</span>`;
    document.querySelector("[data-pipeline-menu]").innerHTML = pipelines.map((pipeline) => `<a href="#/composer/pipeline/${encodeURIComponent(pipeline.key)}"><strong>${esc(pipeline.display_name)}</strong><span>${pipeline.revisions.length} révision${pipeline.revisions.length === 1 ? "" : "s"}</span></a>`).join("");
  }
  function render() {
    const parts = route(); if (parts[0] !== "composer") { location.hash = "#/composer"; return; }
    if (parts[1] === "pipeline" && parts[2]) selectPipeline(decodeURIComponent(parts[2])); else if (!state.pipeline) selectPipeline();
    const section = ["model", "preparation", "search", "response"].includes(parts[1]) ? parts[1] : "composer";
    document.querySelector("[data-view]").innerHTML = renderComposer(section);
    document.querySelectorAll("[data-nav]").forEach((item) => item.classList.toggle("active", item.dataset.nav === section));
    document.querySelectorAll("[data-pipeline-key]").forEach((item) => item.classList.toggle("active", item.dataset.pipelineKey === state.pipeline?.key));
    document.querySelector("[data-active-pipeline]").textContent = state.pipeline?.display_name || "Aucun pipeline";
    document.querySelector("[data-workspace-meta]").innerHTML = state.revision ? `<span>Révision <strong>v${state.revision.revision_number}</strong></span><span>${badge(statusLabel(state.revision.status))}</span><span>Mode <strong>Production / Travail</strong></span>` : "";
  }
  async function executeStep(node) {
    const raw = document.querySelector("[data-step-input]")?.value || "";
    const picked = Array.from(document.querySelectorAll("[data-step-artifact]:checked")).map((item) => item.value);
    const automatic = node.capability.key === "document_extraction" || picked.length || raw.trim() ? [] : compatibleExecutions(node).map((item) => `${item.artifact_type_key}:${item.id}`);
    const lines = [...picked, ...automatic, ...raw.split(/\n/).map((line) => line.trim()).filter(Boolean)];
    const inputs = lines.map((line) => { const separator = line.indexOf(":"); return { artifact_type_key: separator >= 0 ? line.slice(0, separator).trim() : line, artifact_id: separator >= 0 ? line.slice(separator + 1).trim() : "" }; });
    state.stepResults[node.id] = { status: "running", inputs, outputs: [], processing_runs: [] }; render();
    try {
      const response = await fetch("execute-step/", { method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken }, body: JSON.stringify({ pipeline_revision_id: state.revision.id, rag_template_node_id: node.id, document_version_id: state.selectedDocumentId, inputs }) });
      const result = await response.json();
      state.stepResults[node.id] = { ...result, status: result.status || (response.ok ? "completed" : "failed"), message: result.error || result.unsupported_reason || "" };
      if (result.execution_id) { payload.lab_executions = (payload.lab_executions || []).filter((item) => item.execution_id !== result.execution_id); payload.lab_executions.unshift({ ...result, pipeline_revision_id: state.revision.id, rag_template_node_id: node.id, document_version_id: state.selectedDocumentId, input_artifacts: result.input_artifacts || [], output_artifacts: result.output_artifacts || [] }); }
    } catch (_error) { state.stepResults[node.id] = { status: "failed", message: "Le service d’exécution est indisponible.", inputs: [], outputs: [], processing_runs: [] }; }
    render();
  }
  async function postMutation(url, body) {
    const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken }, body: JSON.stringify(body) }); const result = await response.json(); if (!response.ok) throw new Error(result.detail || result.error || "La mutation a échoué."); return result;
  }
  async function refreshProjection(selectedRevisionId, selectedNodeId = null) {
    const response = await fetch("data/", { headers: { Accept: "application/json" } }); const next = await response.json(); if (!response.ok) throw new Error(next.error || "Le rechargement du Composer a échoué.");
    payload = next; pipelines = Array.isArray(next.pipelines) ? next.pipelines : []; const pipelineId = state.pipeline?.id; state.pipeline = pipelines.find((item) => item.id === pipelineId) || pipelines[0] || null; state.revision = state.pipeline?.revisions.find((item) => item.id === selectedRevisionId) || pickRevision(state.pipeline); state.selectedNodeId = selectedNodeId; restoreExecutions(); setupNavigation();
  }
  async function runMutation(work) { if (state.mutationBusy) return; state.mutationBusy = true; state.mutationError = ""; render(); try { await work(); } catch (error) { state.mutationError = error.message; } state.mutationBusy = false; render(); }
  function forkRevision() { const sourceRevision = productionRevision() || state.revision; if (sourceRevision) runMutation(async () => { const result = await postMutation("revisions/fork/", { pipeline_revision_id: sourceRevision.id }); await refreshProjection(result.pipeline_revision_id); }); }
  function emptyRevision() { if (state.revision) runMutation(async () => { const result = await postMutation("revisions/empty/", { pipeline_revision_id: state.revision.id }); await refreshProjection(result.pipeline_revision_id); }); }
  function createNode(form) { const capabilityId = new FormData(form).get("capability_id"); runMutation(async () => { const result = await postMutation("nodes/", { pipeline_revision_id: state.revision.id, capability_id: capabilityId }); state.showAddFunction = false; await refreshProjection(result.pipeline_revision_id, result.rag_template_node_id); }); }
  function selectTool(select) { const nodeId = select.dataset.nodeId; runMutation(async () => { await postMutation("nodes/select-tool/", { pipeline_revision_id: state.revision.id, rag_template_node_id: nodeId, component_version_id: select.value, configuration: {} }); await refreshProjection(state.revision.id, nodeId); }); }
  function createEdge(form) { const nodeId = form.dataset.targetNode; runMutation(async () => { await postMutation("edges/", { pipeline_revision_id: state.revision.id, source_node_id: new FormData(form).get("source_node_id"), target_node_id: nodeId }); await refreshProjection(state.revision.id, nodeId); }); }

  root.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]"); if (action?.dataset.action === "toggle-pipeline-menu") { const menu = document.querySelector("[data-pipeline-menu]"); menu.hidden = !menu.hidden; action.setAttribute("aria-expanded", String(!menu.hidden)); return; }
    if (event.target.closest("[data-close-panel]")) { state.selectedNodeId = null; render(); return; }
    if (event.target.closest("[data-fork-revision]")) { forkRevision(); return; }
    if (event.target.closest("[data-empty-revision]")) { emptyRevision(); return; }
    if (event.target.closest("[data-toggle-add-function]")) { state.showAddFunction = !state.showAddFunction; render(); return; }
    const testButton = event.target.closest("[data-test-node]"); if (testButton) { const node = state.revision.nodes.find((item) => item.id === testButton.dataset.testNode); if (node) executeStep(node); return; }
    const nodeButton = event.target.closest("[data-select-node]"); if (nodeButton) { state.selectedNodeId = nodeButton.dataset.selectNode; render(); }
  });
  root.addEventListener("submit", (event) => { const edge = event.target.closest("[data-edge-form]"); if (edge) { event.preventDefault(); createEdge(edge); return; } const form = event.target.closest("[data-add-function-form]"); if (form) { event.preventDefault(); createNode(form); } });
  root.addEventListener("change", (event) => {
    if (event.target.matches("[data-tool-select]")) { selectTool(event.target); return; }
    if (event.target.matches("[data-document-select]")) { state.selectedDocumentId = event.target.value; localStorage.setItem("kaliok-rag-lab-document", state.selectedDocumentId); restoreExecutions(); render(); return; }
    if (event.target.matches("[data-revision-select]")) { state.revision = state.pipeline.revisions.find((item) => item.id === event.target.value) || state.revision; state.selectedNodeId = null; restoreExecutions(); render(); }
  });
  window.addEventListener("hashchange", render);
  setupNavigation(); selectPipeline(); restoreExecutions(); render();
})();
