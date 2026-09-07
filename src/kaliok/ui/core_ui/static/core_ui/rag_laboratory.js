document.addEventListener("DOMContentLoaded", () => {
  const tabs = [...document.querySelectorAll("[data-tab]")];
  const serverTabs = [...document.querySelectorAll(".tabs a.tab-button")];
  const panels = [...document.querySelectorAll("[data-lab-panel]")];
  function showTab(name) {
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.setAttribute("aria-selected", String(active));
      tab.classList.toggle("is-active", active);
      tab.classList.toggle("is-inactive", !active);
    });
    serverTabs.forEach((tab) => {
      tab.classList.remove("is-active");
      tab.classList.add("is-inactive");
      tab.setAttribute("aria-current", "false");
    });
    panels.forEach((panel) => { panel.hidden = panel.dataset.labPanel !== name; });
  }
  tabs.forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));
  serverTabs.forEach((tab) => tab.addEventListener("click", (event) => {
    event.preventDefault();
    const name = new URL(tab.href, window.location.href).searchParams.get("tab");
    if (name) {
      showTab(name);
      window.history.replaceState({}, "", tab.href);
    }
  }));

  const historicalRoot = document.querySelector('[data-lab-panel="historical_tools"]');
  const historicalPanels = [...(historicalRoot?.querySelectorAll("[data-historical-panel]") || [])];
  const legacyPanels = [...(historicalRoot?.querySelectorAll("[data-panel]") || [])];
  const historicalTools = [...(historicalRoot?.querySelectorAll("[data-historical-tool]") || [])];
  function showHistoricalTool(name) {
    historicalTools.forEach((tool) => {
      const active = tool.dataset.historicalTool === name;
      tool.classList.toggle("is-active", active);
      tool.classList.toggle("is-inactive", !active);
      tool.setAttribute("aria-selected", String(active));
    });
    historicalPanels.forEach((panel) => { panel.hidden = panel.dataset.historicalPanel !== name; });
    legacyPanels.forEach((panel) => {
      if (["results", "history", "settings"].includes(panel.dataset.panel)) {
        panel.hidden = panel.dataset.panel !== name;
      } else if (["discovery", "entity_resolution", "experiment"].includes(panel.dataset.panel)) {
        panel.hidden = false;
      }
    });
  }
  historicalTools.forEach((tool) => tool.addEventListener("click", () => showHistoricalTool(tool.dataset.historicalTool)));
  const initialNavigation = document.querySelector("[data-lab-navigation]");
  showTab(initialNavigation?.dataset.initialTab || "pipeline");
  showHistoricalTool(initialNavigation?.dataset.initialTool || "experiment");

  const revisions = [...document.querySelectorAll('input[name="configuration_revision_ids"]')];
  const repetitions = document.querySelector('#id_repetitions');
  const questionCount = document.querySelector('[data-question-count]');
  const total = document.querySelector('[data-total-runs]');
  const naturalSummary = document.querySelector('[data-natural-summary]');
  function updateTotal() {
    if (!total) return;
    const configurations = revisions.filter((item) => item.checked).length;
    const questions = Number(questionCount?.dataset.questionCount || 0);
    const repeats = Math.max(1, Number(repetitions?.value || 1));
    const runs = configurations * questions * repeats;
    total.textContent = `${questions} question${questions > 1 ? "s" : ""} × ${configurations} configuration${configurations > 1 ? "s" : ""} × ${repeats} essai${repeats > 1 ? "s" : ""} = ${runs} réponse${runs > 1 ? "s" : ""}`;
    if (naturalSummary) naturalSummary.textContent = `Cette expérience va poser ${questions} question${questions > 1 ? "s" : ""} à ${configurations} configuration${configurations > 1 ? "s" : ""}, ${repeats} fois chacune. ${runs} réponse${runs > 1 ? "s seront" : " sera"} générée${runs > 1 ? "s" : ""}.`;
  }
  revisions.forEach((item) => item.addEventListener("change", updateTotal));
  repetitions?.addEventListener("input", updateTotal);
  updateTotal();

  document.querySelector('[data-run-form]')?.addEventListener("submit", (event) => {
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    if (button.disabled) { event.preventDefault(); return; }
    button.disabled = true;
    form.classList.add("is-loading");
  });

  document.getElementById("document-selector")?.addEventListener("change", (event) => {
    const option = event.currentTarget.options[event.currentTarget.selectedIndex];
    const documentInput = document.getElementById("question-document-id");
    if (documentInput) documentInput.value = option.dataset.documentId;
  });

  document.querySelector("[data-feedback-toggle]")?.addEventListener("click", () => {
    const form = document.querySelector("[data-feedback-form]");
    if (form) form.hidden = !form.hidden;
  });

  const versionRows = [...document.querySelectorAll("[data-version-row]")];
  versionRows.forEach((row) => {
    const toggle = row.querySelector("[data-version-toggle]");
    const detail = row.querySelector("[data-version-detail]");
    toggle?.addEventListener("click", () => {
      const opening = detail.hidden;
      versionRows.forEach((otherRow) => {
        const otherDetail = otherRow.querySelector("[data-version-detail]");
        const otherToggle = otherRow.querySelector("[data-version-toggle]");
        if (otherDetail) otherDetail.hidden = true;
        otherToggle?.setAttribute("aria-expanded", "false");
      });
      detail.hidden = !opening;
      toggle.setAttribute("aria-expanded", String(opening));
      if (opening) detail.querySelector("summary, button, input, select")?.focus();
    });
  });

  async function fetchJson(url, signal) {
    const response = await fetch(url, { headers: { Accept: "application/json" }, credentials: "same-origin", signal });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.error) throw new Error(payload.error || "La reponse du laboratoire est invalide.");
    return payload;
  }

  function updateLoadingState(panel, loading) {
    panel.setAttribute("aria-busy", String(loading));
    panel.querySelectorAll("[data-async-discovery-control], [data-async-entity-control], [data-async-discovery-more], [data-async-entity-more]")
      .forEach((control) => { control.disabled = loading; });
    const status = panel.querySelector("[data-async-status]");
    if (status) status.textContent = loading ? "Chargement..." : "";
  }

  function updateUrl(panel, params) {
    const url = new URL(window.location.href);
    Object.entries(params).forEach(([key, value]) => {
      if (value === null || value === "") url.searchParams.delete(key);
      else url.searchParams.set(key, value);
    });
    window.history.replaceState({}, "", url);
  }

  function textElement(tag, text, className) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text == null ? "" : String(text);
    return element;
  }

  function appendTechnicalDetails(parent, pairs) {
    const details = document.createElement("details");
    details.append(textElement("summary", "Details techniques"));
    const list = document.createElement("dl");
    list.className = "technical-list";
    pairs.forEach(([label, value]) => list.append(textElement("dt", label), textElement("dd", value || "Non renseigne")));
    details.append(list);
    parent.append(details);
  }

  function renderOccurrence(candidate) {
    const article = document.createElement("article");
    article.className = "card occurrence-card";
    const head = document.createElement("div");
    head.className = "section-head";
    head.append(textElement("h3", "Page " + (candidate.page_number || "?")), textElement("span", candidate.candidate_type || "", "badge"));
    article.append(head);
    const quote = document.createElement("blockquote");
    quote.append(textElement("span", candidate.highlight_before || ""));
    if (candidate.highlight_text) quote.append(textElement("mark", candidate.highlight_text));
    quote.append(textElement("span", candidate.highlight_after || ""));
    article.append(quote);
    if (candidate.confidence != null) article.append(textElement("p", "Confiance : " + Number(candidate.confidence).toFixed(2), "muted"));
    appendTechnicalDetails(article, [
      ["Methode", (candidate.detector_key || "?") + " v" + (candidate.detector_version || "-")],
      ["Execution Candidate Discovery", candidate.processing_run_id],
      ["Unite normalisee", candidate.normalized_content_unit_id],
      ["Bloc de contenu", candidate.content_block_id],
      ["Fragment", candidate.content_block_fragment_id],
      ["Offsets", (candidate.start_offset ?? "-") + "-" + (candidate.end_offset ?? "-")],
    ]);
    return article;
  }

  function discoveryQuery(panel, offset) {
    const form = panel.querySelector("[data-async-discovery-form]") || panel.querySelector(".discovery-layout form");
    const query = new URLSearchParams({ kind: "discovery", run: panel.dataset.run, limit: "25", offset: String(offset) });
    if (form) new FormData(form).forEach((value, key) => {
      if (["value", "candidate_type", "page"].includes(key) && value) query.set(key, value);
    });
    return query;
  }

  async function loadDiscoveryOccurrences(panel, { append = false, offset = 0 } = {}) {
    const list = panel.querySelector("[data-discovery-occurrences]");
    if (!list || !panel.dataset.run) return;
    if (!append) list.replaceChildren(textElement("p", "Chargement...", "async-placeholder"));
    updateLoadingState(panel, true);
    try {
      const payload = await fetchJson(panel.dataset.asyncUrl + "?" + discoveryQuery(panel, offset));
      if (!append) list.replaceChildren();
      const seen = new Set([...list.querySelectorAll("[data-candidate-id]")].map((item) => item.dataset.candidateId));
      (payload.items || []).forEach((candidate) => {
        const id = String(candidate.candidate_id || "");
        if (!id || !seen.has(id)) {
          const card = renderOccurrence(candidate);
          if (id) card.dataset.candidateId = id;
          list.append(card);
          seen.add(id);
        }
      });
      if (!payload.items?.length && !list.children.length) list.append(textElement("article", "Aucune occurrence pour ces filtres.", "card"));
      const more = panel.querySelector("[data-async-discovery-more]");
      if (more) {
        more.hidden = offset + (payload.items || []).length >= Number(payload.total || 0);
        more.dataset.offset = String(offset + (payload.items || []).length);
      }
      const count = panel.querySelector(".visible-count");
      if (count) count.textContent = list.querySelectorAll(".occurrence-card").length + " occurrence(s) affichee(s) sur " + (payload.total || 0);
      updateUrl(panel, { tab: "discovery", discovery_run: panel.dataset.run, value: panel.querySelector('[name="value"]')?.value || "", candidate_type: panel.querySelector('[name="candidate_type"]')?.value || "", page: panel.querySelector('[name="page"]')?.value || "" });
    } catch (error) {
      list.replaceChildren(textElement("article", "Erreur : " + error.message, "card"));
    } finally {
      updateLoadingState(panel, false);
    }
  }

  function renderEntitySummary(summary, detail) {
    const article = document.createElement("article");
    article.className = "card entity-card";
    const head = document.createElement("div");
    head.className = "section-head";
    const identity = document.createElement("div");
    identity.append(textElement("p", "Entite proposee", "eyebrow"), textElement("h3", summary.canonical_label));
    identity.append(textElement("span", summary.entity_type || "", "badge"));
    head.append(identity);
    const counts = document.createElement("div");
    counts.className = "entity-counts";
    counts.append(textElement("strong", (summary.membership_count || 0) + " occurrence(s)"), textElement("span", (summary.document_count || 0) + " document(s) - " + (summary.page_count || 0) + " page(s)"));
    head.append(counts);
    article.append(head);
    if (!detail) article.append(textElement("p", "Cliquez pour charger la provenance complete.", "muted"));
    if (detail) {
      article.append(textElement("p", summary.membership_count > 1 ? "Pourquoi ce rapprochement ? Meme type et meme valeur normalisee." : "1 occurrence - aucun rapprochement avec une autre occurrence.", "resolution-reason"));
      const memberships = document.createElement("section");
      memberships.className = "membership-list";
      memberships.append(textElement("h4", "Occurrences"));
      (detail.memberships || []).forEach((membership) => {
        const occurrence = document.createElement("article");
        occurrence.className = "occurrence-card";
        occurrence.append(textElement("strong", membership.document?.filename || "Document inconnu"));
        (membership.provenance || []).forEach((source) => {
          const quote = document.createElement("blockquote");
          quote.append(textElement("span", source.highlight_before || ""));
          if (source.highlight_text) quote.append(textElement("mark", source.highlight_text));
          quote.append(textElement("span", source.highlight_after || ""));
          occurrence.append(quote);
        });
        appendTechnicalDetails(occurrence, [["Membership UUID", membership.membership_id], ["Candidate UUID", membership.candidate?.candidate_id], ["Valeur normalisee", membership.candidate?.normalized_value]]);
        memberships.append(occurrence);
      });
      article.append(memberships);
    }
    return article;
  }

  function entityQuery(panel, offset, entity) {
    const form = panel.querySelector("[data-async-entity-form]") || panel.querySelector(".entity-resolution-filters");
    const query = new URLSearchParams({ kind: "entity_resolution", run: panel.dataset.run, limit: "25", offset: String(offset) });
    if (form) new FormData(form).forEach((value, key) => {
      if (["canonical_label", "entity_type"].includes(key) && value) query.set(key, value);
    });
    if (entity) query.set("entity", entity);
    return query;
  }

  async function loadEntityResolutionDetail(panel, { append = false, offset = 0, entity = "" } = {}) {
    const list = panel.querySelector("[data-entity-list]");
    if (!list || !panel.dataset.run) return;
    updateLoadingState(panel, true);
    try {
      const payload = await fetchJson(panel.dataset.asyncUrl + "?" + entityQuery(panel, offset, entity));
      if (!append) list.replaceChildren();
      const existing = new Set([...list.querySelectorAll("[data-entity-id]")].map((item) => item.dataset.entityId));
      (payload.items || []).forEach((summary) => {
        const id = String(summary.id);
        if (!existing.has(id)) {
          const card = renderEntitySummary(summary, id === entity ? payload.detail : null);
          card.dataset.entityId = id;
          list.append(card);
          existing.add(id);
        }
      });
      if (!payload.items?.length && !list.children.length) list.append(textElement("article", "Aucune entite proposee pour ces filtres.", "card"));
      const more = panel.querySelector("[data-async-entity-more]");
      if (more) {
        more.hidden = offset + (payload.items || []).length >= Number(payload.total || 0);
        more.dataset.offset = String(offset + (payload.items || []).length);
      }
      const count = panel.querySelector(".visible-count");
      if (count) count.textContent = list.querySelectorAll("[data-entity-id]").length + " entite(s) affichee(s) sur " + (payload.total || 0);
      updateUrl(panel, { tab: "entity_resolution", entity_resolution_run: panel.dataset.run, entity: entity || "" });
    } catch (error) {
      list.replaceChildren(textElement("article", "Erreur : " + error.message, "card"));
    } finally {
      updateLoadingState(panel, false);
    }
  }

  document.querySelectorAll("[data-async-discovery-control='run']").forEach((control) => control.addEventListener("change", () => {
    const panel = control.closest("[data-panel]");
    panel.dataset.run = control.value;
    loadDiscoveryOccurrences(panel);
  }));
  document.querySelectorAll("[data-async-discovery-form]").forEach((form) => form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadDiscoveryOccurrences(form.closest("[data-panel]"));
  }));
  document.querySelectorAll("[data-async-discovery-more]").forEach((control) => control.addEventListener("click", (event) => {
    event.preventDefault();
    const panel = control.closest("[data-panel]");
    loadDiscoveryOccurrences(panel, { append: true, offset: Number(control.dataset.offset || 25) });
  }));
  document.querySelectorAll(".discovery-group").forEach((control) => control.addEventListener("click", (event) => {
    event.preventDefault();
    const url = new URL(control.href, window.location.href);
    const panel = control.closest("[data-panel]");
    ["value", "candidate_type"].forEach((key) => {
      const field = panel.querySelector('[name="' + key + '"]');
      if (field) field.value = url.searchParams.get(key) || "";
    });
    loadDiscoveryOccurrences(panel);
  }));
  document.querySelectorAll("[data-async-entity-control='run']").forEach((control) => control.addEventListener("change", () => {
    const panel = control.closest("[data-panel]");
    panel.dataset.run = control.value;
    loadEntityResolutionDetail(panel);
  }));
  document.addEventListener("change", (event) => {
    const control = event.target;
    if (!(control instanceof HTMLSelectElement)) return;
    if (control.name !== "discovery_document" && control.name !== "discovery_run" && control.name !== "entity_resolution_run") return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const panel = control.closest("[data-panel]");
    if (!panel) return;
    if (control.name === "discovery_document") panel.dataset.run = control.selectedOptions[0]?.dataset.run || "";
    else panel.dataset.run = control.value;
    if (control.name === "entity_resolution_run") loadEntityResolutionDetail(panel);
    else if (control.name === "discovery_run") loadDiscoveryOccurrences(panel);
  }, true);
  document.querySelectorAll("[data-async-entity-form]").forEach((form) => form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadEntityResolutionDetail(form.closest("[data-panel]"));
  }));
  document.querySelectorAll("[data-async-entity-more]").forEach((control) => control.addEventListener("click", (event) => {
    event.preventDefault();
    const panel = control.closest("[data-panel]");
    loadEntityResolutionDetail(panel, { append: true, offset: Number(control.dataset.offset || 25) });
  }));
  document.querySelectorAll(".entity-picker a").forEach((control) => control.addEventListener("click", (event) => {
    event.preventDefault();
    const panel = control.closest("[data-panel]");
    const url = new URL(control.href, window.location.href);
    loadEntityResolutionDetail(panel, { entity: url.searchParams.get("entity") || "" });
  }));

  const pipelineRoot = document.querySelector("[data-pipeline-root]");
  const pipelineSelector = pipelineRoot?.querySelector("[data-pipeline-document]");
  const pipelineStatus = pipelineRoot?.querySelector("[data-pipeline-status]");
  const configStatus = pipelineRoot?.querySelector("[data-pipeline-config-status]");
  const pipelineRunButton = pipelineRoot?.querySelector("[data-pipeline-run]");
  let pipelineSelection = [];
  const pipelineRoleLabels = Object.freeze({
    document_extraction: "Lecture du document",
    normalization: "Structuration du contenu",
    entity_discovery: "Découverte d’entités",
    entity_resolution: "Résolution d’entités",
    chunking: "Découpage sémantique",
    indexing: "Indexation",
  });

  function pipelineRole(capability) {
    return pipelineRoleLabels[capability] || capability;
  }

  function componentTitle(component) {
    if (!component?.component_key) return "Aucun intervenant sélectionné";
    if (component.display_name) return component.display_name;
    const label = component.component_key
      .replace(/^kaliok-/, "Kaliok ")
      .replace(/^postgres-/, "PostgreSQL ")
      .replace(/-/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
    return `${label} ${component.version || component.component_version || ""}`.trim();
  }

  function groupByComponent(items, getComponent) {
    const groups = new Map();
    items.forEach((item) => {
      const component = getComponent(item);
      const key = component?.component_key ? `${component.component_key}@@${component.component_version || component.version}` : `none@@${item.key || item.capability}`;
      if (!groups.has(key)) groups.set(key, { component, items: [] });
      groups.get(key).items.push(item);
    });
    return [...groups.values()];
  }

  function initializePipelineSelection() {
    const grouped = new Map();
    pipelineRoot?.querySelectorAll("[data-capability-select]").forEach((select, index) => {
      if (!select.value) return;
      const [component_key, component_version] = select.value.split("@@");
      const identity = `${component_key}@@${component_version}`;
      let binding = grouped.get(identity);
      if (!binding) {
        binding = { binding_key: `binding-${index + 1}`, component_key, component_version, capabilities: [], configuration: {}, dependencies: [], enabled: true };
        grouped.set(identity, binding);
      }
      if (!binding.capabilities.includes(select.dataset.capability)) binding.capabilities.push(select.dataset.capability);
    });
    pipelineSelection = [...grouped.values()];
  }

  function pipelineText(parent, tag, value, className) {
    const node = textElement(tag, value == null ? "" : value, className);
    parent.append(node);
    return node;
  }

  function csrfToken() {
    const value = document.cookie.split(";").map((item) => item.trim()).find((item) => item.startsWith("csrftoken="))?.split("=").slice(1).join("=") || "";
    return decodeURIComponent(value);
  }

  function selectionForPayload() {
    return pipelineSelection.map((binding) => ({
      binding_key: binding.binding_key,
      component_key: binding.component_key,
      component_version: binding.component_version,
      capabilities: [...(binding.capabilities || [])],
      configuration: binding.configuration || {},
      dependencies: [...(binding.dependencies || [])],
      enabled: binding.enabled !== false,
    }));
  }

  function renderPipelineManifest(container, manifest, title, eyebrow, editable = false) {
    if (!container || !manifest) return;
    container.replaceChildren();
    pipelineText(container, "p", eyebrow, "eyebrow");
    pipelineText(container, "h3", title);
    pipelineText(container, "p", manifest.description, "help");
    pipelineText(container, "p", manifest.revision, "badge");
    (manifest.bindings || []).forEach((binding) => {
      const row = document.createElement("div");
      row.className = "pipeline-binding";
      pipelineText(row, "strong", `${componentTitle({ component_key: binding.component_key, version: binding.component_version })}${binding.enabled === false ? " · désactivé" : ""}`);
      pipelineText(row, "span", (binding.role_labels || binding.capabilities || []).map((role) => typeof role === "string" && pipelineRoleLabels[role] ? pipelineRole(role) : role).join(" · "));
      pipelineText(row, "small", "Rôles couverts par cet intervenant");
      const details = document.createElement("details");
      pipelineText(details, "summary", "Configuration");
      if (editable) {
        pipelineText(details, "p", "Éditeur JSON validé côté serveur.", "help");
        if (binding.definition?.configuration_schema && Object.keys(binding.definition.configuration_schema).length) {
          appendTechnicalDetails(details, [["configuration_schema", JSON.stringify(binding.definition.configuration_schema)]]);
        }
        const textarea = document.createElement("textarea");
        textarea.dataset.pipelineConfig = binding.binding_key;
        textarea.rows = 5;
        textarea.value = JSON.stringify(binding.configuration || {}, null, 2);
        details.append(textarea);
        const save = document.createElement("button");
        save.type = "button";
        save.className = "secondary";
        save.dataset.pipelineConfigSave = binding.binding_key;
        pipelineText(save, "span", "Valider la configuration");
        details.append(save);
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "secondary";
        toggle.dataset.pipelineToggle = binding.binding_key;
        pipelineText(toggle, "span", binding.enabled === false ? "Activer le binding" : "Désactiver le binding");
        details.append(toggle);
      } else pipelineText(details, "pre", JSON.stringify(binding.configuration || {}, null, 2));
      appendTechnicalDetails(details, [
        ["binding_key", binding.binding_key],
        ["component_key", binding.component_key],
        ["component_version", binding.component_version],
        ["capabilities", (binding.capabilities || []).join(", ")],
      ]);
      row.append(details);
      container.append(row);
    });
    const details = document.createElement("details");
    pipelineText(details, "summary", "Détails manifest");
    appendTechnicalDetails(details, [["pipeline_key", manifest.pipeline_key], ["revision", manifest.revision], ["manifest_hash", manifest.manifest_hash]]);
    container.append(details);
  }

  function renderPipelineCapabilities(state) {
    const container = pipelineRoot?.querySelector("[data-pipeline-capabilities]");
    if (!container) return;
    container.replaceChildren();
    const selected = (state.capabilities || []).filter((capability) => capability.selected_component);
    const unselected = (state.capabilities || []).filter((capability) => !capability.selected_component);
    const groups = groupByComponent(selected, (capability) => capability.selected_component);
    groups.forEach((group, index) => {
      const article = document.createElement("article");
      article.className = "pipeline-capability";
      article.dataset.capabilityRow = "";
      article.dataset.component = group.component.component_key;
      const head = document.createElement("div");
      head.className = "pipeline-capability-head";
      const name = document.createElement("div");
      pipelineText(name, "strong", componentTitle(group.component));
      pipelineText(name, "span", `Étape ${index + 1}`, "help");
      pipelineText(head, "span", group.items.map((item) => item.status).every((status) => status === "EXÉCUTABLE") ? "EXÉCUTABLE" : group.items[0].status, "status-label");
      head.prepend(name);
      article.append(head);
      const roles = document.createElement("ul");
      roles.className = "pipeline-role-list";
      group.items.forEach((capability) => {
        const role = document.createElement("li");
        pipelineText(role, "strong", capability.role_label || pipelineRole(capability.key));
        roles.append(role);
      });
      article.append(roles);
      pipelineText(article, "p", `Binding ${group.items[0].selected_binding_key || "—"}`, "help");
      const chooser = document.createElement("details");
      pipelineText(chooser, "summary", "Modifier les rôles");
      group.items.forEach((capability) => {
        const label = document.createElement("label");
        pipelineText(label, "span", capability.role_label || pipelineRole(capability.key));
        const select = document.createElement("select");
        select.dataset.capabilitySelect = "";
        select.dataset.capability = capability.key;
        (capability.components || []).forEach((component) => {
          const option = document.createElement("option");
          option.value = `${component.component_key}@@${component.version}`;
          option.textContent = `${componentTitle(component)} · ${component.runtime_status}`;
          if (capability.selected_component.component_key === component.component_key && capability.selected_component.component_version === component.version) option.selected = true;
          select.append(option);
        });
        label.append(select);
        chooser.append(label);
      });
      article.append(chooser);
      appendTechnicalDetails(article, [["capabilities", group.items.map((item) => item.key).join(", ")]]);
      container.append(article);
    });
    unselected.forEach((capability, index) => {
      const article = document.createElement("article");
      article.className = "pipeline-capability pipeline-capability-unselected";
      const head = document.createElement("div");
      head.className = "pipeline-capability-head";
      const name = document.createElement("div");
      pipelineText(name, "strong", "Aucun intervenant sélectionné");
      pipelineText(name, "span", `Étape ${selected.length + index + 1} · ${pipelineRole(capability.key)}`, "help");
      head.append(name);
      pipelineText(head, "span", capability.status, "status-label");
      article.append(head);
      const label = document.createElement("label");
      pipelineText(label, "span", "Choisir l’intervenant");
      const select = document.createElement("select");
      select.dataset.capabilitySelect = "";
      select.dataset.capability = capability.key;
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "Aucun intervenant sélectionné";
      select.append(empty);
      (capability.components || []).forEach((component) => {
        const option = document.createElement("option");
        option.value = `${component.component_key}@@${component.version}`;
        option.textContent = `${componentTitle(component)} · ${component.runtime_status}`;
        select.append(option);
      });
      label.append(select);
      article.append(label);
      container.append(article);
    });
  }

  function renderPipelineStages(container, stages) {
    if (!container) return;
    container.replaceChildren();
    groupByComponent(stages || [], (stage) => stage.component).forEach((group) => {
      const row = document.createElement("div");
      row.className = "pipeline-stage";
      const identity = document.createElement("div");
      pipelineText(identity, "strong", componentTitle(group.component));
      pipelineText(identity, "span", group.items.map((stage) => stage.role_label || pipelineRole(stage.capability)).join(" · "), "help");
      const run = document.createElement("div");
      group.items.forEach((stage) => {
        const last = stage.last_run;
        pipelineText(run, "span", stage.role_label || pipelineRole(stage.capability));
        pipelineText(run, "small", last ? `Dernier run : ${last.status} · ${last.artifact_count ?? "—"} artefact(s)` : "Aucun run chargé");
        if (last?.execution_environment === "experiment" && last.execution_group_id) {
        const inspect = document.createElement("button");
        inspect.type = "button";
        inspect.className = "secondary";
        inspect.dataset.pipelineInspect = stage.key;
        inspect.dataset.pipelineGroup = last.execution_group_id;
        pipelineText(inspect, "span", stage.key === "entity_discovery" ? "Inspecter les candidats" : "Inspecter");
        run.append(inspect);
        }
      });
      row.append(identity, run);
      appendTechnicalDetails(row, [["capabilities", group.items.map((stage) => stage.capability).join(", ")]]);
      container.append(row);
    });
  }

  function renderPipelineHistory(container, history) {
    if (!container) return;
    container.replaceChildren();
    if (!history?.length) return pipelineText(container, "p", "Aucune exécution expérimentale disponible.", "muted");
    history.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-card pipeline-history-button";
      button.dataset.pipelineHistory = item.execution_group_id;
      pipelineText(button, "span", `${item.started_at || "Date inconnue"} · ${item.pipeline_key} / ${item.revision}`);
      pipelineText(button, "span", item.status, `badge ${item.status}`);
      container.append(button);
    });
  }

  function formatDuration(value) {
    return value == null || Number(value) <= 0 ? "non mesurée" : `${value} ms`;
  }

  function renderPipelineInspection(parent, inspection) {
    const section = document.createElement("div");
    section.className = "pipeline-inspection";
    const start = Number(inspection.offset || 0) + 1;
    const end = Number(inspection.offset || 0) + (inspection.items || []).length;
    pipelineText(section, "p", `Résultats ${inspection.total ? `${start}–${end} sur ${inspection.total}` : "0"} · Page source : ${Math.floor((inspection.offset || 0) / (inspection.limit || 25)) + 1}`, "help");
    (inspection.items || []).forEach((item) => {
      const card = document.createElement("article");
      card.className = "choice";
      if (inspection.kind === "perception") {
        pipelineText(card, "strong", `Page ${item.page} · ${item.type}`);
        pipelineText(card, "p", item.content);
        pipelineText(card, "small", `${item.method || "Méthode inconnue"} · ${item.engine || "Moteur inconnu"}${item.confidence == null ? "" : ` · confiance ${item.confidence}`}`, "muted");
        appendTechnicalDetails(card, [["bbox", item.bbox ? JSON.stringify(item.bbox) : "Non renseignée"], ["ContentBlock UUID", item.id]]);
      } else if (inspection.kind === "discovery") {
        pipelineText(card, "strong", `${item.raw_value || item.normalized_value || "Candidat"} · ${item.candidate_type}`);
        pipelineText(card, "p", item.exact_text || item.raw_value || "");
        pipelineText(card, "small", `Page ${item.page_number} · ${item.detector_key}@${item.detector_version}${item.confidence == null ? "" : ` · confiance ${item.confidence}`}`, "muted");
        appendTechnicalDetails(card, [
          ["NormalizedContentUnit UUID", item.normalized_content_unit_id],
          ["ContentBlock UUID", item.content_block_id],
          ["Fragment UUID", item.content_block_fragment_id],
          ["Offsets", `${item.start_offset}–${item.end_offset}`],
          ["Source", item.unit_content],
        ]);
      } else {
        pipelineText(card, "strong", `Unité ${item.order} · ${item.content_type}`);
        pipelineText(card, "p", item.content);
        (item.sources || []).forEach((source) => {
          pipelineText(card, "small", `Source · page ${source.page} · bloc ${source.block_id}`, "muted");
          pipelineText(card, "blockquote", source.content || "");
        });
      }
      section.append(card);
    });
    if (inspection.offset + inspection.items.length < inspection.total) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "secondary";
      more.dataset.pipelineMore = inspection.kind;
      more.dataset.pipelineOffset = String(inspection.offset + inspection.items.length);
      pipelineText(more, "span", "Charger la suite");
      section.append(more);
    }
    parent.append(section);
  }

  function renderPipelineRun(parent, run, label) {
    const section = document.createElement("section");
    section.className = "pipeline-run-detail";
    pipelineText(section, "h4", label);
    const metrics = document.createElement("div");
    metrics.className = "metrics";
    [["Statut", run?.status || "non lancé"], ["Durée", formatDuration(run?.duration_ms)], ["Moteur", run ? `${run.engine || "—"}@${run.engine_version || "—"}` : "—"], ["Artefacts", run?.artifact_count ?? "—"]].forEach(([name, value]) => {
      const metric = document.createElement("div");
      metric.className = "metric";
      pipelineText(metric, "span", name);
      pipelineText(metric, "strong", value);
      metrics.append(metric);
    });
    section.append(metrics);
    if (run?.error) pipelineText(section, "p", "Cette étape a échoué. Les résultats disponibles restent inspectables.", "alert error");
    if (run) {
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "secondary";
      const lowerLabel = label.toLowerCase();
      const inspectKind = lowerLabel.startsWith("perception") ? "perception" : lowerLabel.startsWith("discovery") ? "discovery" : "normalization";
      inspect.dataset.pipelineInspect = inspectKind;
      pipelineText(inspect, "span", inspectKind === "perception" ? "Inspecter les ContentBlocks" : inspectKind === "discovery" ? "Inspecter les candidats" : "Inspecter les NormalizedContentUnits");
      section.append(inspect);
      if (run.inspection) renderPipelineInspection(section, run.inspection);
      appendTechnicalDetails(section, [["ProcessingRun UUID", run.id], ["execution_environment", run.execution_environment], ["execution_group_id", run.execution_group_id], ["configuration", JSON.stringify(run.configuration || {})], ["metrics", JSON.stringify(run.metrics || {})]]);
    }
    parent.append(section);
  }

  function renderPipelineResult(root, state) {
    const container = document.querySelector("[data-pipeline-result]");
    if (!container) return;
    const result = state.result;
    container.replaceChildren();
    pipelineText(container, "p", "Résultats Pipeline_A courant", "eyebrow");
    if (!result) {
      pipelineText(container, "h3", "Prêt à inspecter");
      pipelineText(container, "p", "Lancez les étapes exécutables pour voir les ProcessingRuns, ContentBlocks et NormalizedContentUnits.", "muted");
      return;
    }
    pipelineText(container, "h3", `Pipeline_A · ${result.status}`);
    pipelineText(container, "p", `${result.document?.filename || "Document"} · durée totale ${formatDuration(result.duration_ms)}`);
    const runs = document.createElement("div");
    runs.className = "pipeline-runs";
    renderPipelineRun(runs, result.perception, "Perception");
    renderPipelineRun(runs, result.normalization, "Normalisation");
    if (result.discovery) renderPipelineRun(runs, result.discovery, "Discovery");
    container.append(runs);
    appendTechnicalDetails(container, [["execution_group_id", result.execution_group_id]]);
    if (state.execution_error) pipelineText(container, "p", state.execution_error, "alert error");
    if (state.technical_error) appendTechnicalDetails(container, [["Erreur technique", state.technical_error]]);
  }

  function renderPipelineState(state) {
    if (!pipelineRoot || !state) return;
    pipelineSelection = state.pipeline_selection || [];
    pipelineRoot.dataset.version = state.selected_document?.document_version_id || "";
    pipelineRoot.dataset.group = state.result?.execution_group_id || "";
    if (pipelineSelector) {
      const current = state.selected_document?.document_version_id || "";
      const options = document.createDocumentFragment();
      (state.documents || []).forEach((documentItem) => {
        const option = document.createElement("option");
        option.value = documentItem.document_version_id;
        option.dataset.executable = String(Boolean(documentItem.executable));
        option.selected = documentItem.document_version_id === current;
        option.textContent = `${documentItem.filename} · v${documentItem.version_number} · ${documentItem.processing_status}${documentItem.page_count ? ` · ${documentItem.page_count} page(s)` : ""}`;
        options.append(option);
      });
      pipelineSelector.replaceChildren(options);
      pipelineSelector.disabled = !(state.documents || []).length;
    }
    const documentMeta = pipelineRoot.querySelector("[data-pipeline-document-meta]");
    if (documentMeta && state.selected_document) documentMeta.textContent = `${state.selected_document.title || state.selected_document.filename} · hash ${state.selected_document.file_hash_short || "—"}`;
    const documentWarning = pipelineRoot.querySelector("[data-pipeline-document-warning]");
    if (documentWarning) {
      documentWarning.hidden = Boolean(state.selected_document?.executable);
      documentWarning.textContent = state.selected_document ? "Cette version reste visible mais ne peut pas être lancée : le runtime n’a pas de nombre de pages persistant." : "Aucun document exploitable disponible.";
    }
    renderPipelineCapabilities(state);
    renderPipelineManifest(pipelineRoot.querySelector("[data-pipeline-reference]"), state.pipeline_reference, "Pipeline_P", "Pipeline de référence");
    renderPipelineManifest(pipelineRoot.querySelector("[data-pipeline-experiment]"), state.pipeline_experiment, "Pipeline_A", "Manifest courant · construit par le serveur", true);
    renderPipelineStages(pipelineRoot.querySelector("[data-pipeline-stages]"), state.stages);
    renderPipelineHistory(document.querySelector("[data-pipeline-history]"), state.history);
    renderPipelineResult(pipelineRoot, state);
    if (pipelineRunButton) pipelineRunButton.disabled = !state.selected_document?.executable;
  }

  async function loadPipelineState(extra = {}) {
    if (!pipelineRoot) return null;
    pipelineRoot.setAttribute("aria-busy", "true");
    if (pipelineStatus) pipelineStatus.textContent = "Chargement ciblé…";
    try {
      const params = new URLSearchParams(extra);
      if (!params.has("document_version_id")) params.set("document_version_id", pipelineRoot.dataset.version || pipelineSelector?.value || "");
      params.set("selection", JSON.stringify(selectionForPayload()));
      const state = await fetchJson(`${pipelineRoot.dataset.pipelineUrl}?${params}`);
      renderPipelineState(state);
      if (pipelineStatus) pipelineStatus.textContent = !state.selected_document ? "Aucun document exploitable disponible." : state.selected_document.executable ? "" : "Cette DocumentVersion est visible mais non exécutable pour le runtime.";
      return state;
    } catch (error) {
      if (pipelineStatus) pipelineStatus.textContent = `Erreur : ${error.message}`;
      return null;
    } finally { pipelineRoot.setAttribute("aria-busy", "false"); }
  }

  async function configurePipeline() {
    if (!pipelineRoot) return;
    if (configStatus) configStatus.textContent = "Validation du manifest…";
    try {
      const response = await fetch(pipelineRoot.dataset.pipelineUrl, {
        method: "POST", credentials: "same-origin",
        headers: { Accept: "application/json", "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
        body: JSON.stringify({ action: "configure", document_version_id: pipelineSelector?.value || "", selection: selectionForPayload() }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.error) throw new Error(payload.error || "Le manifest Pipeline_A est invalide.");
      renderPipelineState(payload);
      if (configStatus) configStatus.textContent = "Pipeline_A validé et reconstruit côté serveur.";
    } catch (error) { if (configStatus) configStatus.textContent = `Configuration refusée : ${error.message}`; }
  }

  function changeCapability(capability, value) {
    const kept = [];
    pipelineSelection.forEach((binding) => {
      binding.capabilities = (binding.capabilities || []).filter((item) => item !== capability);
      if (binding.capabilities.length) kept.push(binding);
    });
    if (value) {
      const [component_key, component_version] = value.split("@@");
      let target = pipelineSelection.find((binding) => binding.component_key === component_key && binding.component_version === component_version);
      if (!target) {
        target = { binding_key: `binding-${kept.length + 1}`, component_key, component_version, capabilities: [], configuration: {}, dependencies: [], enabled: true };
        kept.push(target);
      }
      target.capabilities.push(capability);
    }
    pipelineSelection = kept;
    configurePipeline();
  }

  async function executePipeline() {
    if (!pipelineRoot || !pipelineSelector?.value || !pipelineRunButton || pipelineRunButton.disabled) return;
    pipelineRunButton.disabled = true;
    if (pipelineStatus) pipelineStatus.textContent = "Exécution… les bindings non raccordés seront signalés.";
    try {
      const response = await fetch(pipelineRoot.dataset.pipelineUrl, { method: "POST", credentials: "same-origin", headers: { Accept: "application/json", "Content-Type": "application/json", "X-CSRFToken": csrfToken() }, body: JSON.stringify({ document_version_id: pipelineSelector.value, selection: selectionForPayload() }) });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.error) throw new Error(payload.error || "L’exécution du pipeline a échoué.");
      renderPipelineState(payload);
      if (pipelineStatus) pipelineStatus.textContent = payload.execution_error || "Exécution terminée. Les résultats sont persistés comme expérience.";
    } catch (error) { if (pipelineStatus) pipelineStatus.textContent = `Erreur : ${error.message}`; }
    finally { pipelineRunButton.disabled = !pipelineSelector?.selectedOptions[0]?.dataset.executable || false; }
  }

  pipelineSelector?.addEventListener("change", () => loadPipelineState({ document_version_id: pipelineSelector.value }));
  pipelineRunButton?.addEventListener("click", executePipeline);
  pipelineRoot?.addEventListener("change", (event) => {
    const select = event.target.closest("[data-capability-select]");
    if (select) changeCapability(select.dataset.capability, select.value);
  });
  document.addEventListener("click", async (event) => {
    if (!pipelineRoot || (!pipelineRoot.contains(event.target) && !event.target.closest("[data-pipeline-result], [data-pipeline-history]"))) return;
    const historyButton = event.target.closest("[data-pipeline-history]");
    const inspectButton = event.target.closest("[data-pipeline-inspect]");
    const moreButton = event.target.closest("[data-pipeline-more]");
    const configSave = event.target.closest("[data-pipeline-config-save]");
    const toggleButton = event.target.closest("[data-pipeline-toggle]");
    if (configSave) {
      const binding = pipelineSelection.find((item) => item.binding_key === configSave.dataset.pipelineConfigSave);
      const textarea = pipelineRoot.querySelector(`[data-pipeline-config="${CSS.escape(configSave.dataset.pipelineConfigSave)}"]`);
      if (!binding || !textarea) return;
      try {
        const parsed = JSON.parse(textarea.value || "{}");
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("un objet JSON est attendu");
        binding.configuration = parsed;
        await configurePipeline();
      } catch (error) { if (configStatus) configStatus.textContent = `Configuration refusée : ${error.message}`; }
      return;
    }
    if (toggleButton) {
      const binding = pipelineSelection.find((item) => item.binding_key === toggleButton.dataset.pipelineToggle);
      if (binding) { binding.enabled = binding.enabled === false; await configurePipeline(); }
      return;
    }
    const group = inspectButton?.dataset.pipelineGroup || pipelineRoot.dataset.group || "";
    if (historyButton) await loadPipelineState({ document_version_id: pipelineRoot.dataset.version, execution_group_id: historyButton.dataset.pipelineHistory });
    else if (inspectButton) await loadPipelineState({ document_version_id: pipelineRoot.dataset.version, execution_group_id: group, inspect: inspectButton.dataset.pipelineInspect, offset: "0", limit: "25" });
    else if (moreButton) await loadPipelineState({ document_version_id: pipelineRoot.dataset.version, execution_group_id: group, inspect: moreButton.dataset.pipelineMore, offset: moreButton.dataset.pipelineOffset, limit: "25" });
  });
  initializePipelineSelection();
  if (pipelineRoot) loadPipelineState();
});
