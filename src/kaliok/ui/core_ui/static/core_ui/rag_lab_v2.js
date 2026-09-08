document.addEventListener("DOMContentLoaded", () => {
  const root = document.querySelector("[data-rag-lab-root]");
  if (!root) return;

  let state = null;
  let selection = [];
  let dirty = false;
  let replacementKey = "";
  const roleLabels = {
    document_extraction: "Lecture du document",
    normalization: "Structuration du contenu",
    entity_discovery: "Découverte d’entités",
    entity_resolution: "Résolution d’entités",
    chunking: "Découpage sémantique",
    indexing: "Indexation",
  };
  const text = (value) => value == null ? "" : String(value);
  const label = (key) => roleLabels[key] || key;
  const apiUrl = root.dataset.pipelineUrl;
  const csrf = () => document.cookie.split(";").map((item) => item.trim()).find((item) => item.startsWith("csrftoken="))?.split("=").slice(1).join("=") || "";

  function element(tag, value, className) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== undefined) item.textContent = text(value);
    return item;
  }

  async function request(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", headers: { Accept: "application/json", ...(options.headers || {}) }, ...options });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.error) throw new Error(payload.error || "Réponse invalide du Lab.");
    return payload;
  }

  function setStatus(message, target = "[data-pipeline-status]") {
    const item = document.querySelector(target);
    if (item) item.textContent = message || "";
  }

  function activateTab(name) {
    document.querySelectorAll("[data-tab]").forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("is-active", active);
      tab.classList.toggle("is-inactive", !active);
      tab.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll("[data-lab-panel]").forEach((panel) => { panel.hidden = panel.dataset.labPanel !== name; });
  }

  function technical(parent, pairs) {
    const details = document.createElement("details");
    details.append(element("summary", "Détails techniques"));
    const list = element("dl", undefined, "technical-list");
    pairs.forEach(([key, value]) => { list.append(element("dt", key), element("dd", value || "—")); });
    details.append(list);
    parent.append(details);
  }

  function componentTitle(item) {
    if (item.display_name) return item.display_name;
    const catalogItem = (state?.catalogue || []).find((candidate) => candidate.component_key === item.component_key && (candidate.version === item.version || candidate.version === item.component_version));
    return catalogItem?.display_name || `${item.component_key} ${item.version || item.component_version || ""}`.trim();
  }

  function componentFor(binding) {
    return (state?.catalogue || []).find((item) => item.component_key === binding.component_key && item.version === binding.component_version);
  }

  function availableCapabilities(component, chosen) {
    const capabilities = component?.capabilities || [];
    const selected = new Set(chosen || []);
    const byBundle = new Map();
    capabilities.forEach((item) => { if (item.execution_bundle_key) byBundle.set(item.execution_bundle_key, [...(byBundle.get(item.execution_bundle_key) || []), item]); });
    byBundle.forEach((members) => {
      if (members.some((item) => item.invocation_mode === "all_or_none" && selected.has(item.key))) members.forEach((item) => selected.add(item.key));
      members.filter((item) => item.invocation_mode === "produced_with_bundle").forEach((item) => {
        if (!members.some((member) => member.invocation_mode !== "produced_with_bundle" && selected.has(member.key))) selected.delete(item.key);
      });
    });
    return [...selected].filter((key) => capabilities.some((item) => item.key === key));
  }

  function bindingCard(binding, editable) {
    const card = element("article", undefined, `pipeline-binding-card${editable ? " editable-binding" : ""}`);
    card.dataset.bindingKey = binding.binding_key;
    const index = element("div", undefined, "binding-index");
    index.textContent = String(selection.findIndex((item) => item.binding_key === binding.binding_key) + 1);
    const main = element("div", undefined, "binding-main");
    const component = componentFor(binding);
    main.append(element("strong", componentTitle(binding)), element("span", (binding.capabilities || []).map(label).join(" · ")), element("small", component?.runtime_status || "—"));
    card.append(index, main);
    if (editable) {
      const actions = element("div", undefined, "binding-actions");
      [["configure", "Configurer", ""], ["replace", "Remplacer", ""], ["remove", "Retirer", "danger-text"]].forEach(([action, title, className]) => {
        const button = element("button", title, `link-button ${className}`);
        button.type = "button"; button.dataset.bindingAction = action; button.dataset.bindingKey = binding.binding_key; actions.append(button);
      });
      card.append(actions);
    }
    if (editable) {
      const details = document.createElement("details");
      details.append(element("summary", "Configuration et rôles"));
      const roles = element("div", undefined, "binding-role-editor");
      (component?.capabilities || []).forEach((capability) => {
        const role = document.createElement("label");
        role.className = "role-option";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox"; checkbox.checked = (binding.capabilities || []).includes(capability.key); checkbox.dataset.bindingRole = binding.binding_key; checkbox.dataset.roleKey = capability.key;
        checkbox.disabled = capability.invocation_mode === "produced_with_bundle";
        role.append(checkbox, element("span", `${capability.role_label || label(capability.key)} · ${capability.invocation_mode}${capability.execution_bundle_key ? ` · ${capability.execution_bundle_key}` : ""}`));
        roles.append(role);
      });
      details.append(roles);
      const configuration = document.createElement("textarea");
      configuration.rows = 3; configuration.value = JSON.stringify(binding.configuration || {}, null, 2); configuration.dataset.bindingConfiguration = binding.binding_key; configuration.setAttribute("aria-label", `Configuration de ${componentTitle(binding)}`);
      details.append(configuration);
      const saveConfiguration = element("button", "Valider la configuration", "secondary");
      saveConfiguration.type = "button"; saveConfiguration.dataset.configurationSave = binding.binding_key; details.append(saveConfiguration);
      technical(details, [["binding_key", binding.binding_key], ["component_key", binding.component_key], ["version", binding.component_version], ["rôles utilisés", (binding.capabilities || []).join(", ")], ["dépendances", (binding.dependencies || []).join(", ")] ]);
      card.append(details);
    } else {
      technical(card, [["binding_key", binding.binding_key], ["component_key", binding.component_key], ["version", binding.component_version], ["rôles utilisés", (binding.capabilities || []).join(", ")] ]);
    }
    return card;
  }

  function renderBindings() {
    const draft = document.querySelector("[data-draft-bindings]");
    if (draft) { draft.replaceChildren(...selection.map((binding) => bindingCard(binding, true))); }
    const production = document.querySelector("[data-pipeline-production] .binding-list");
    if (production && state?.pipeline_reference) production.replaceChildren(...(state.pipeline_reference.bindings || []).map((binding) => bindingCard(binding, false)));
  }

  function renderSummary() {
    const template = state?.template || {};
    const missing = template.missing_required || [];
    const composition = document.querySelector("[data-composition-status]");
    if (composition) composition.textContent = `Composition : ${missing.length ? "INCOMPLET" : "VALIDE"}`;
    const message = document.querySelector("[data-composition-message]");
    if (message) message.textContent = missing.length ? `Fonction requise non couverte : ${missing.map((item) => item.role_label).join(", ")}` : "Toutes les fonctions requises du template sont couvertes.";
    const runtime = document.querySelector("[data-runtime-status]");
    const activeIdentities = new Set(selection.map((item) => `${item.component_key}@${item.component_version}`));
    const summary = {
      binding_count: selection.length,
      executable_count: selection.filter((item) => componentFor(item)?.runtime_executable).length,
      catalogue_unwired_count: (state?.catalogue || []).filter((item) => !activeIdentities.has(`${item.component_key}@${item.version}`)).length,
    };
    if (runtime) runtime.textContent = summary.binding_count === summary.executable_count ? "Exécutable" : "Partiellement exécutable";
    const runtimeMeta = document.querySelector("[data-draft-state] .runtime-summary small");
    if (runtimeMeta) runtimeMeta.textContent = `${summary.binding_count || 0} intervenant(s) · ${summary.executable_count || 0} exécutable(s) · ${summary.catalogue_unwired_count || 0} catalogué(s) non raccordé(s)`;
    const executeSummary = document.querySelector("[data-execute-summary]");
    if (executeSummary) {
      executeSummary.replaceChildren(element("strong", `Pipeline utilisé : Gestation · Révision ${state?.draft_revision?.revision_number || "—"}`), element("span", selection.map((item) => componentTitle(item)).join(" · ") || "Aucun intervenant"));
    }
  }

  function renderCatalogue(items = state?.catalogue || []) {
    const list = document.querySelector("[data-catalogue-list]");
    if (!list) return;
    list.replaceChildren(...items.map((component) => {
      const card = element("article", undefined, "catalogue-card");
      const head = element("div", undefined, "section-head");
      const identity = element("div"); identity.append(element("h3", componentTitle(component)), element("p", component.version, "muted"));
      head.append(identity, element("span", component.runtime_status, "status-label")); card.append(head);
      if (component.description) card.append(element("p", component.description));
      card.append(element("p", `Capabilities fournies : ${(component.capabilities || []).map((item) => item.role_label || label(item.key)).join(" · ") || "Aucune"}`));
      const roles = element("div", undefined, "catalogue-capabilities");
      (component.capabilities || []).forEach((item) => { const role = element("span", undefined, "catalogue-capability"); role.append(element("strong", item.role_label || label(item.key)), element("small", `${item.invocation_mode}${item.execution_bundle_key ? ` · bundle ${item.execution_bundle_key}` : ""}`)); roles.append(role); });
      card.append(roles); technical(card, [["component_key", component.component_key], ["runtime status", component.runtime_status], ["configuration_schema", JSON.stringify(component.configuration_schema || {})]]); return card;
    }));
  }

  function renderAddDrawer() {
    const drawer = document.querySelector("[data-component-drawer]");
    const list = document.querySelector("[data-add-component-list]");
    if (!drawer || !list) return;
    const used = new Set(selection.map((item) => `${item.component_key}@${item.component_version}`));
    const replaceUsed = replacementKey ? new Set(selection.filter((item) => item.binding_key !== replacementKey).map((item) => `${item.component_key}@${item.component_version}`)) : used;
    const items = (state?.catalogue || []).filter((item) => !replaceUsed.has(`${item.component_key}@${item.version}`));
    list.replaceChildren(...items.map((component) => {
      const card = element("article", undefined, "catalogue-card add-card");
      card.append(element("h4", componentTitle(component)), element("p", component.version, "muted"), element("p", `Runtime : ${component.runtime_status}`));
      const roles = element("div", undefined, "binding-role-editor");
      (component.capabilities || []).forEach((role) => { const item = element("span", `${role.role_label || label(role.key)} · ${role.invocation_mode}${role.execution_bundle_key ? ` · ${role.execution_bundle_key}` : ""}`, "role-option"); roles.append(item); });
      const add = element("button", replacementKey ? "Remplacer par cet intervenant" : "Ajouter cet intervenant"); add.type = "button"; add.dataset.addIdentity = `${component.component_key}@@${component.version}`; card.append(roles, add); return card;
    }));
    drawer.hidden = false;
  }

  function addComponent(component) {
    const capabilities = availableCapabilities(component, (component.capabilities || []).map((item) => item.key));
    const binding = { binding_key: replacementKey || `binding-${selection.length + 1}`, component_key: component.component_key, component_version: component.version, capabilities, configuration: {}, dependencies: [], enabled: true, display_name: componentTitle(component) };
    if (replacementKey) selection = selection.map((item) => item.binding_key === replacementKey ? binding : item); else selection.push(binding);
    replacementKey = ""; dirty = true; document.querySelector("[data-component-drawer]")?.setAttribute("hidden", "");
    renderBindings(); renderSummary(); setStatus("Modification locale. Enregistrez la draft pour la persister.");
  }

  function updateRoles(bindingKey, roleKey, checked) {
    const binding = selection.find((item) => item.binding_key === bindingKey); const component = componentFor(binding || {}); if (!binding || !component) return;
    const next = new Set(binding.capabilities || []); const role = (component.capabilities || []).find((item) => item.key === roleKey);
    if (checked) next.add(roleKey); else next.delete(roleKey);
    if (!checked && role?.invocation_mode === "all_or_none" && role.execution_bundle_key) (component.capabilities || []).filter((item) => item.execution_bundle_key === role.execution_bundle_key).forEach((item) => next.delete(item.key));
    binding.capabilities = availableCapabilities(component, [...next]);
    dirty = true; renderBindings(); renderSummary();
  }

  function renderHistory() {
    const list = document.querySelector("[data-pipeline-inspect-history]"); if (!list) return;
    list.replaceChildren(...(state?.history || []).map((item) => { const button = element("button", undefined, "history-card pipeline-history-button"); button.type = "button"; button.dataset.pipelineHistory = item.execution_group_id; button.append(element("span", `${item.started_at || "Date inconnue"} · Gestation / ${item.revision || "—"}`), element("span", item.status, `badge ${item.status}`)); return button; }));
    if (!state?.history?.length) list.append(element("p", "Aucune exécution expérimentale disponible.", "muted"));
  }

  function renderInspection(run, kind, inspection) {
    const section = element("section", undefined, "pipeline-inspection");
    (inspection?.items || []).forEach((item) => { const card = element("article", undefined, "choice"); const title = kind === "perception" ? `ContentBlock · page ${item.page}` : kind === "normalization" ? `NormalizedContentUnit · ${item.order}` : kind === "discovery" ? `${item.candidate_type || "Candidat"} · ${item.normalized_value || item.raw_value || ""}` : `${item.canonical_label || "Entity"} · ${item.entity_type || ""}`; card.append(element("strong", title), element("p", item.content || item.unit_content || item.raw_value || item.canonical_label || "")); technical(card, [["id", item.id], ["run", run.id]]); section.append(card); });
    if (inspection && inspection.offset + inspection.items.length < inspection.total) { const more = element("button", "Charger la suite", "secondary"); more.type = "button"; more.dataset.pipelineMore = kind; more.dataset.pipelineOffset = String(inspection.offset + inspection.items.length); section.append(more); }
    return section;
  }

  function renderResult(result) {
    const target = document.querySelector("[data-pipeline-result]"); if (!target) return; target.replaceChildren(element("p", "Étapes réellement exécutées", "eyebrow"));
    if (!result) { target.append(element("h2", "Prêt à inspecter"), element("p", "Sélectionnez une exécution dans la liste.", "muted")); return; }
    target.append(element("h2", `Gestation · ${result.status}`), element("p", `${result.document?.filename || "Document"} · durée totale ${result.duration_ms || "—"} ms`));
    if (state?.execution_error) target.append(element("p", state.execution_error, "alert error"));
    if (state?.technical_error) technical(target, [["Erreur technique", state.technical_error]]);
    const steps = [["perception", "Reader", result.perception], ["normalization", "Normalizer", result.normalization], ["discovery", "Discovery", result.discovery], ["resolution", "Resolution", result.resolution]];
    const list = element("div", undefined, "pipeline-stage-list");
    steps.filter(([, , run]) => run).forEach(([kind, name, run]) => { const row = element("article", undefined, "pipeline-stage"); const info = element("div"); info.append(element("strong", name), element("small", `${run.status} · ${run.artifact_count ?? "—"} artefact(s)`)); const actions = element("div"); const button = element("button", `Inspecter ${name}`, "secondary"); button.type = "button"; button.dataset.pipelineInspectKind = kind; button.dataset.pipelineGroup = result.execution_group_id; actions.append(button); if (run.inspection) actions.append(renderInspection(run, kind, run.inspection)); row.append(info, actions); list.append(row); }); target.append(list); technical(target, [["execution_group_id", result.execution_group_id], ["pipeline_revision_id", state?.draft_revision?.id]]);
  }

  function renderCompare(kind) {
    const target = document.querySelector("[data-compare-result]"); if (!target) return; target.replaceChildren();
    if (kind === "runs") { const comparison = state?.result?.comparison; if (!comparison?.available) { target.append(element("p", comparison?.message || "Sélectionnez un run inspectable.", "muted")); return; } target.append(element("h3", "Run de production vs run de gestation"), element("pre", JSON.stringify(comparison.delta || {}, null, 2))); return; }
    const comparison = state?.comparison || {}; target.append(element("h3", kind === "revision" ? "Révision N vs N+1" : "Production vs gestation")); const changes = [...(comparison.components?.added || []).map((item) => `+ ${item.component_key}@${item.component_version}`), ...(comparison.components?.removed || []).map((item) => `− ${item.component_key}@${item.component_version}`), ...Object.keys(comparison.binding_changes || {}).map((key) => `~ ${key}`)]; target.append(element("p", changes.length ? changes.join(" · ") : "Aucune différence de composition.")); technical(target, [["pipeline_p", JSON.stringify(comparison.pipeline_p || {})], ["pipeline_a", JSON.stringify(comparison.pipeline_a || {})]]);
  }

  async function load(extra = {}) {
    const params = new URLSearchParams(extra); const version = document.querySelector("[data-execute-document]")?.value || ""; if (!params.has("document_version_id") && version) params.set("document_version_id", version);
    try { state = await request(`${apiUrl}?${params}`); selection = (state.pipeline_selection || []).map((item) => ({ ...item, capabilities: [...(item.capabilities || [])], dependencies: [...(item.dependencies || [])], configuration: { ...(item.configuration || {}) } })); dirty = false; root.dataset.version = state.selected_document?.document_version_id || ""; renderBindings(); renderSummary(); renderCatalogue(); renderHistory(); renderResult(state.result); return state; } catch (error) { setStatus(`Erreur : ${error.message}`); return null; }
  }

  async function persist(action) {
    setStatus(action === "save" ? "Enregistrement…" : "Mise à jour de la révision…");
    try { const payload = await request(apiUrl, { method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": csrf() }, body: JSON.stringify({ action, document_version_id: root.dataset.version || "", selection }) }); state = payload; selection = (state.pipeline_selection || []).map((item) => ({ ...item, capabilities: [...(item.capabilities || [])], configuration: { ...(item.configuration || {}) } })); dirty = false; renderBindings(); renderSummary(); renderCatalogue(); renderHistory(); setStatus(action === "save" ? "Brouillon enregistré dans PostgreSQL." : "Révision mise à jour."); } catch (error) { setStatus(`Enregistrement refusé : ${error.message}`); }
  }

  document.querySelectorAll("[data-tab]").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
  document.querySelector("[data-add-component]")?.addEventListener("click", () => { replacementKey = ""; renderAddDrawer(); });
  document.querySelector("[data-close-drawer]")?.addEventListener("click", () => document.querySelector("[data-component-drawer]")?.setAttribute("hidden", ""));
  document.querySelector("[data-pipeline-save]")?.addEventListener("click", () => persist("save"));
  document.querySelector("[data-pipeline-reset]")?.addEventListener("click", () => persist("from_production"));
  document.querySelector("[data-pipeline-new-revision]")?.addEventListener("click", () => persist("new_revision"));
  document.querySelector("[data-pipeline-cancel]")?.addEventListener("click", () => load());
  document.querySelector("[data-execute-document]")?.addEventListener("change", (event) => load({ document_version_id: event.target.value }));
  document.querySelector("[data-execute-run]")?.addEventListener("click", async () => { if (dirty) { setStatus("Enregistrez la draft avant de lancer.", "[data-execute-status]"); return; } const button = document.querySelector("[data-execute-run]"); button.disabled = true; setStatus("Exécution…", "[data-execute-status]"); try { state = await request(apiUrl, { method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": csrf() }, body: JSON.stringify({ document_version_id: document.querySelector("[data-execute-document]")?.value || root.dataset.version }) }); renderHistory(); renderResult(state.result); activateTab("inspect"); setStatus(state.execution_error || "Exécution terminée.", "[data-execute-status]"); } catch (error) { setStatus(`Erreur : ${error.message}`, "[data-execute-status]"); } finally { button.disabled = false; } });
  document.addEventListener("click", async (event) => {
    const action = event.target.closest("[data-binding-action]");
    if (action) { const key = action.dataset.bindingKey; if (action.dataset.bindingAction === "remove") { selection = selection.filter((item) => item.binding_key !== key); dirty = true; renderBindings(); renderSummary(); setStatus("Intervenant retiré localement. Enregistrez la draft."); } else if (action.dataset.bindingAction === "replace") { replacementKey = key; renderAddDrawer(); } else { const details = action.closest(".pipeline-binding-card")?.querySelector("details"); if (details) details.open = true; } return; }
    const add = event.target.closest("[data-add-identity]"); if (add) { const [componentKey, version] = add.dataset.addIdentity.split("@@"); const component = (state.catalogue || []).find((item) => item.component_key === componentKey && item.version === version); if (component) addComponent(component); return; }
    const role = event.target.closest("[data-binding-role]"); if (role) { updateRoles(role.dataset.bindingRole, role.dataset.roleKey, role.checked); return; }
    const config = event.target.closest("[data-configuration-save]"); if (config) { const binding = selection.find((item) => item.binding_key === config.dataset.configurationSave); const field = document.querySelector(`[data-binding-configuration="${CSS.escape(config.dataset.configurationSave)}"]`); try { const value = JSON.parse(field?.value || "{}"); if (!value || Array.isArray(value)) throw new Error("un objet JSON est attendu"); binding.configuration = value; dirty = true; setStatus("Configuration locale validée. Enregistrez la draft."); } catch (error) { setStatus(`Configuration refusée : ${error.message}`); } return; }
    const history = event.target.closest("[data-pipeline-history]"); if (history) { const result = await load({ execution_group_id: history.dataset.pipelineHistory }); activateTab("inspect"); renderResult(result?.result); return; }
    const inspect = event.target.closest("[data-pipeline-inspect-kind]"); if (inspect) { const result = await load({ execution_group_id: inspect.dataset.pipelineGroup, inspect: inspect.dataset.pipelineInspectKind, offset: "0", limit: "25" }); renderResult(result?.result); return; }
    const more = event.target.closest("[data-pipeline-more]"); if (more) { const result = await load({ execution_group_id: root.dataset.group || "", inspect: more.dataset.pipelineMore, offset: more.dataset.pipelineOffset, limit: "25" }); renderResult(result?.result); return; }
    const compare = event.target.closest("[data-compare]"); if (compare) renderCompare(compare.dataset.compare);
  });

  activateTab(document.querySelector("[data-lab-navigation]")?.dataset.initialTab || "composer");
  load();
});
