(function () {
  "use strict";

  const bundle = window.KNETMINER_BUNDLE;
  const elements = {
    search: document.getElementById("case-search"),
    list: document.getElementById("case-list"),
    summary: document.getElementById("case-summary"),
    filters: Array.from(document.querySelectorAll("[data-filter]")),
    allCount: document.getElementById("all-count"),
    answeredCount: document.getElementById("answered-count"),
    abstainedCount: document.getElementById("abstained-count"),
    kicker: document.getElementById("case-kicker"),
    title: document.getElementById("case-title"),
    status: document.getElementById("answer-status"),
    answer: document.getElementById("answer-text"),
    pathCount: document.getElementById("path-count"),
    paths: document.getElementById("path-list"),
    provenance: document.getElementById("provenance-grid")
  };
  const state = { filter: "all", query: "", selected: null };

  function appendText(parent, tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = value;
    parent.appendChild(node);
    return node;
  }

  function answerEntries() {
    return Object.entries(bundle.answers).sort(([left], [right]) => left.localeCompare(right));
  }

  function visibleEntries() {
    return answerEntries().filter(([questionId, answer]) => {
      const matchesStatus = state.filter === "all" || answer.status === state.filter;
      const searchable = [questionId, answer.intent].concat(answer.entity_ids).join(" ").toLowerCase();
      return matchesStatus && searchable.includes(state.query);
    });
  }

  function renderCases() {
    const entries = visibleEntries();
    elements.list.replaceChildren();
    elements.summary.textContent = `${entries.length} matching ${entries.length === 1 ? "case" : "cases"}`;

    entries.forEach(([questionId, answer]) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = `case-button${questionId === state.selected ? " is-active" : ""}`;
      button.dataset.questionId = questionId;
      button.setAttribute("aria-current", questionId === state.selected ? "true" : "false");

      const label = document.createElement("span");
      appendText(label, "strong", "", questionId);
      appendText(label, "small", "", `${answer.intent} / ${answer.entity_ids.join(", ")}`);
      button.appendChild(label);
      appendText(button, "span", `case-dot${answer.status === "abstained" ? " is-abstained" : ""}`, "");
      button.addEventListener("click", () => selectCase(questionId));
      item.appendChild(button);
      elements.list.appendChild(item);
    });

    if (entries.length && !entries.some(([questionId]) => questionId === state.selected)) {
      selectCase(entries[0][0]);
    }
  }

  function inferNodeType(nodeId) {
    if (nodeId.startsWith("ENSG")) return "target";
    if (nodeId.startsWith("CHEMBL")) return "drug";
    return "disease";
  }

  function renderPath(pathId) {
    const path = bundle.evidence_paths[pathId];
    const record = document.createElement("article");
    record.className = "path-record";
    appendText(record, "p", "path-id", pathId);

    const chain = document.createElement("div");
    chain.className = "node-chain";
    path.node_ids.forEach((nodeId, index) => {
      appendText(chain, "span", `node ${inferNodeType(nodeId)}`, nodeId);
      if (index < path.relations.length) appendText(chain, "span", "relation", path.relations[index]);
    });
    record.appendChild(chain);

    const citations = document.createElement("div");
    citations.className = "citation-row";
    appendText(citations, "span", "section-label", "Forward citation edge IDs");
    path.citation_edge_ids.forEach((edgeId) => appendText(citations, "span", "edge-id", edgeId));
    record.appendChild(citations);
    return record;
  }

  function renderPaths(answer) {
    elements.paths.replaceChildren();
    elements.pathCount.textContent = `${answer.citations.length} ${answer.citations.length === 1 ? "path" : "paths"}`;
    if (!answer.citations.length) {
      appendText(
        elements.paths,
        "p",
        "empty-paths",
        "No observed path satisfied this case. The system abstained instead of constructing evidence."
      );
      return;
    }
    answer.citations.forEach((pathId) => elements.paths.appendChild(renderPath(pathId)));
  }

  function renderProvenance() {
    const labels = {
      source_release: "Open Targets release",
      licence: "Data licence",
      model_revision: "Pinned model revision",
      evaluation_run_id: "Evaluation run",
      snapshot_hash: "Snapshot SHA-256",
      evaluation_report_hash: "Evaluation SHA-256"
    };
    elements.provenance.replaceChildren();
    Object.entries(labels).forEach(([key, label]) => {
      const group = document.createElement("div");
      appendText(group, "dt", "", label);
      appendText(group, "dd", "", bundle.provenance[key]);
      elements.provenance.appendChild(group);
    });
  }

  function selectCase(questionId) {
    const answer = bundle.answers[questionId];
    state.selected = questionId;
    elements.kicker.textContent = answer.intent.replaceAll("_", " ");
    elements.title.textContent = `${questionId}: ${answer.entity_ids.join(" + ")}`;
    elements.status.textContent = answer.status;
    elements.status.className = `status-badge${answer.status === "abstained" ? " is-abstained" : ""}`;
    elements.answer.textContent = answer.text || answer.abstention_reason;
    renderPaths(answer);
    renderProvenance();
    renderCases();
  }

  function initialize() {
    if (!bundle || !bundle.answers || !bundle.evidence_paths || !bundle.provenance) {
      elements.title.textContent = "Frozen bundle unavailable";
      elements.answer.textContent = "The release artifact could not be loaded.";
      elements.status.textContent = "Error";
      elements.status.className = "status-badge is-abstained";
      return;
    }

    const entries = answerEntries();
    const answered = entries.filter(([, answer]) => answer.status === "answered").length;
    elements.allCount.textContent = String(entries.length);
    elements.answeredCount.textContent = String(answered);
    elements.abstainedCount.textContent = String(entries.length - answered);

    elements.search.addEventListener("input", (event) => {
      state.query = event.target.value.trim().toLowerCase();
      renderCases();
    });
    elements.filters.forEach((button) => {
      button.addEventListener("click", () => {
        state.filter = button.dataset.filter;
        elements.filters.forEach((candidate) => {
          const active = candidate === button;
          candidate.classList.toggle("is-active", active);
          candidate.setAttribute("aria-pressed", String(active));
        });
        renderCases();
      });
    });

    state.selected = entries[0][0];
    renderCases();
    selectCase(state.selected);
  }

  initialize();
}());
