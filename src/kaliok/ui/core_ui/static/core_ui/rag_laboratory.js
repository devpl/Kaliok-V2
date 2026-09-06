document.addEventListener("DOMContentLoaded", () => {
  const tabs = [...document.querySelectorAll("[data-tab]")];
  const serverTabs = [...document.querySelectorAll(".tabs a.tab-button")];
  const panels = [...document.querySelectorAll("[data-panel]")];
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
    panels.forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
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
});
