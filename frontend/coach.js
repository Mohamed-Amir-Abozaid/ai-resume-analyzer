/*
 * Career & Interview Coach - frontend (HAI features)
 *
 * Plugs into the existing ResumePilot shell. It reuses API_URL from script.js and
 * touches nothing outside the #coach view. All text is inserted with textContent
 * (never innerHTML), so model output cannot inject markup.
 */
(function () {
  "use strict";

  const section = document.querySelector('[data-section="coach"]');
  if (!section) return;

  const MAX_HISTORY = 6;
  const REQUEST_TIMEOUT_MS = 60000;
  const ONBOARDING_KEY = "coachOnboardingDone";

  const TYPE_LABEL = {
    feedback: "Feedback",
    clarification: "A question for you",
    info: "Information",
    out_of_scope: "Outside what I can do",
    refusal: "I can't help with this",
    privacy: "Privacy check"
  };
  const LEVEL_LABEL = { strong: "Strong", developing: "Developing", needs_work: "Needs work" };
  const FOCUS_LABEL = { clarity: "Clarity", relevance: "Relevance", evidence: "Evidence" };
  const STAGE_TEXT = [
    "",
    "Step 1 of 3: watch the coach work on an example.",
    "Step 2 of 3: try it yourself. The coach guides you.",
    "Step 3 of 3: you work on your own. Help stays available."
  ];

  const $ = (selector) => section.querySelector(selector);
  const log = $("#coach-log");
  const form = $("#coach-form");
  const input = $("#coach-input");
  const questionInput = $("#coach-question");
  const jobInput = $("#coach-jd");
  const plainBox = $("#coach-plain");
  const alwaysWhyBox = $("#coach-always-why");
  const sendButton = $("#coach-send");
  const statusEl = $("#coach-status");
  const countEl = $("#coach-count");
  const hintEl = $("#coach-input-hint");
  const focusFieldset = $("#coach-focus");
  const onboarding = $("#coach-onboarding");
  const onboardingBody = $("#coach-onboarding-body");
  const onboardingToggle = $('[data-coach-action="onboarding-toggle"]');
  const personaBox = $("#coach-personas");
  const stepStatus = $("#coach-step-status");
  const steps = [...section.querySelectorAll("[data-step]")];

  const state = {
    greeting: null,
    history: [],
    lastRequest: null,
    busy: false,
    stage: 1,
    persona: null,
    counter: 0,
    started: false
  };
  const messageData = new WeakMap();

  /* ---------- small helpers ---------- */

  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === false || value === null || value === undefined) continue;
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function setStatus(kind, text) {
    const prefix = { ready: "Ready", working: "Working", done: "Done", error: "Error" }[kind];
    statusEl.dataset.state = kind;
    statusEl.textContent = `${prefix}: ${text}`;
  }

  function scrollTo(node) {
    if (node && node.scrollIntoView) node.scrollIntoView({ block: "nearest" });
  }

  function selectedFocus() {
    const checked = focusFieldset.querySelector('input[name="coach-focus"]:checked');
    return checked ? checked.value : "clarity";
  }

  function setFocusValue(value) {
    const radio = focusFieldset.querySelector(`input[value="${value}"]`);
    if (radio) radio.checked = true;
  }

  function updateCount() {
    countEl.textContent = `${input.value.length} / ${input.maxLength} characters`;
  }

  function setBusy(busy) {
    state.busy = busy;
    sendButton.disabled = busy;
    sendButton.textContent = busy ? "Sending…" : "Send";
    focusFieldset.disabled = busy;
    log.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function detailText(data, fallback) {
    const detail = data && data.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length) return "Please check your input. " + (detail[0].msg || "");
    return fallback;
  }

  async function api(path, options) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(`${API_URL}${path}`, {
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        ...(options || {})
      });
      let data = null;
      try { data = await response.json(); } catch (_) { /* not JSON */ }
      if (!response.ok) throw new Error(detailText(data, "Something went wrong. Please try again."));
      return data;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("The request took too long. Please try again. Your text is still in the box.");
      }
      if (error instanceof TypeError) {
        throw new Error("I can't reach the server. Check that the backend is running and your connection works, then try again. Your text is still in the box.");
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  /* ---------- messages ---------- */

  function appendNotice(kind, text, retry) {
    const node = h(
      "div",
      { class: `coach-notice coach-notice-${kind}`, role: kind === "error" ? "alert" : null },
      h("strong", null, kind === "error" ? "Error: " : "Note: "),
      text,
      retry ? h("button", { type: "button", class: "coach-btn", onclick: retry }, "Try again") : null
    );
    log.append(node);
    scrollTo(node);
  }

  function appendUser(text, label) {
    const node = h(
      "article",
      { class: "coach-msg coach-msg-user", "aria-label": "Your message" },
      h("p", { class: "coach-msg-who" }, label || "You"),
      h("p", { class: "coach-msg-text", dir: "auto" }, text)
    );
    log.append(node);
    scrollTo(node);
  }

  function appendGuide(text) {
    const node = h(
      "article",
      { class: "coach-msg coach-msg-bot", "aria-label": "Guide from the AI coach" },
      h("p", { class: "coach-msg-who" }, h("span", { class: "coach-badge" }, "AI coach"), h("span", { class: "coach-tag" }, "Guide")),
      h("p", { class: "coach-msg-text" }, text)
    );
    log.append(node);
    scrollTo(node);
  }

  function renderGreeting(g) {
    const node = h(
      "article",
      { class: "coach-msg coach-msg-bot", "aria-label": "First message from the AI coach" },
      h("p", { class: "coach-msg-who" }, h("span", { class: "coach-badge" }, "AI coach"), h("span", { class: "coach-tag" }, "Welcome")),
      h("p", { class: "coach-msg-text" }, g.message),
      h(
        "div",
        { class: "coach-limits" },
        h("div", null, h("h3", null, "What I can do"), h("ul", null, g.can_do.map((t) => h("li", null, t)))),
        h("div", null, h("h3", null, "What I can't do"), h("ul", null, g.cannot_do.map((t) => h("li", null, t))))
      ),
      h("p", { class: "coach-note" }, g.privacy_note)
    );
    log.append(node);
  }

  function buildWhy(resp, id) {
    const parts = [];

    parts.push(h("h4", null, "Evidence from your text"));
    if (resp.evidence.length) {
      parts.push(h("ul", { class: "coach-evidence" }, resp.evidence.map((q) => h("li", { dir: "auto" }, `“${q}”`))));
    } else if (resp.type === "feedback") {
      parts.push(h("p", null, "I don't have enough evidence from your text for this feedback, so treat it as general advice."));
    } else {
      parts.push(h("p", null, "No quotes from your text were used for this reply."));
    }

    parts.push(h("h4", null, `Rubric criterion (focus: ${FOCUS_LABEL[resp.focus] || resp.focus})`));
    parts.push(h("p", { dir: "auto" }, resp.criterion));
    if (resp.level) {
      parts.push(h("p", null, `Practice level: ${LEVEL_LABEL[resp.level]}. This is for practice only. It is not a hiring score.`));
    }

    parts.push(h("h4", null, "Why I said this"));
    parts.push(h("p", { dir: "auto" }, resp.explanation));

    parts.push(h("h4", null, "Limits"));
    parts.push(h("p", { dir: "auto" }, resp.limitations));

    return h("div", { class: "coach-why", id, hidden: true }, parts);
  }

  function appendBot(resp, request, options) {
    const whyId = `coach-why-${++state.counter}`;
    const canSimplify = !resp.plain_language && resp.type !== "privacy";

    const header = h(
      "p",
      { class: "coach-msg-who" },
      h("span", { class: "coach-badge" }, "AI coach"),
      h("span", { class: "coach-tag" }, TYPE_LABEL[resp.type] || "Reply"),
      resp.type === "feedback" && resp.level ? h("span", { class: "coach-tag" }, `Level: ${LEVEL_LABEL[resp.level]}`) : null,
      h("span", { class: "coach-tag" }, `Focus: ${FOCUS_LABEL[resp.focus] || resp.focus}`),
      resp.plain_language ? h("span", { class: "coach-tag" }, "Plain language") : null,
      options && options.rerunLabel ? h("span", { class: "coach-tag" }, options.rerunLabel) : null
    );

    const node = h(
      "article",
      { class: "coach-msg coach-msg-bot", "data-response-id": resp.response_id, "aria-label": "AI coach reply" },
      header,
      h("div", { class: "coach-msg-text", dir: "auto", lang: resp.language }, resp.answer),
      resp.warnings.length
        ? h("div", { class: "coach-warning" }, resp.warnings.map((w) => h("p", null, h("strong", null, "Note: "), w)))
        : null,
      resp.next_action ? h("p", { class: "coach-next", dir: "auto" }, h("strong", null, "Next step: "), resp.next_action) : null,
      h(
        "div",
        { class: "coach-actions-row", role: "group", "aria-label": "Actions for this reply" },
        h("button", { type: "button", class: "coach-chip", "data-coach-action": "why", "aria-expanded": "false", "aria-controls": whyId }, "Why this?"),
        canSimplify ? h("button", { type: "button", class: "coach-chip", "data-coach-action": "simpler" }, "Simpler explanation") : null,
        h("button", { type: "button", class: "coach-chip", "data-coach-action": "up", "aria-pressed": "false" }, "Helpful"),
        h("button", { type: "button", class: "coach-chip", "data-coach-action": "down", "aria-pressed": "false" }, "Not helpful"),
        h("button", { type: "button", class: "coach-chip", "data-coach-action": "challenge" }, "Challenge this")
      ),
      buildWhy(resp, whyId),
      h("div", { class: "coach-inline-slot" }),
      h("p", { class: "coach-inline-status", role: "status" })
    );

    messageData.set(node, { resp, request });
    log.append(node);
    if ((options && options.forceWhy) || (alwaysWhyBox && alwaysWhyBox.checked)) toggleWhy(node, true);
    scrollTo(node);
    return node;
  }

  function toggleWhy(node, forceOpen) {
    const button = node.querySelector('[data-coach-action="why"]');
    const panel = node.querySelector(".coach-why");
    const open = forceOpen === undefined ? panel.hidden : forceOpen;
    panel.hidden = !open;
    button.setAttribute("aria-expanded", open ? "true" : "false");
  }

  /* ---------- sending ---------- */

  function readRequest(message) {
    return {
      message,
      job_description: jobInput.value.trim(),
      interview_question: questionInput.value.trim(),
      focus: selectedFocus(),
      plain_language: plainBox.checked,
      language: "auto",
      history: state.history.slice(-MAX_HISTORY)
    };
  }

  async function send(request, options) {
    options = options || {};
    if (state.busy) return false;
    setBusy(true);
    const working = options.rerunLabel
      ? `${options.rerunLabel}. Re-running your last message…`
      : "The AI coach is working. This can take a few seconds.";
    setStatus("working", working);
    const pending = h("p", { class: "coach-pending", "aria-hidden": "true" }, "AI coach is writing a reply…");
    log.append(pending);
    scrollTo(pending);

    try {
      const resp = await api("/coach/chat", { method: "POST", body: JSON.stringify(request) });
      pending.remove();

      const history = request.history.concat(
        { role: "user", content: request.message.slice(0, 2000) },
        { role: "assistant", content: resp.answer.slice(0, 2000) }
      );
      state.history = history.slice(-MAX_HISTORY);
      state.lastRequest = request;

      appendBot(resp, request, options);
      if (options.clearInput && input.value.trim() === request.message) {
        input.value = "";
        updateCount();
      }
      setStatus("done", "Reply ready. Choose “Why this?” to see the evidence and the rubric behind it.");
      advanceStage(options);
      return true;
    } catch (error) {
      pending.remove();
      appendNotice("error", error.message, () => send(request, { ...options, retried: true }));
      setStatus("error", error.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  function rerun(overrides, label) {
    if (!state.lastRequest) return;
    const request = { ...state.lastRequest, ...overrides };
    send(request, { rerunLabel: label });
  }

  /* ---------- feedback and challenge (HAI level 4) ---------- */

  function inlineStatus(node, text) {
    node.querySelector(".coach-inline-status").textContent = text;
  }

  function closeInlineForm(node, returnFocus) {
    const slot = node.querySelector(".coach-inline-slot");
    slot.replaceChildren();
    if (returnFocus) returnFocus.focus();
  }

  function openInlineForm(node, kind, trigger) {
    closeInlineForm(node);
    const { resp } = messageData.get(node);
    const id = `coach-inline-${++state.counter}`;
    const isChallenge = kind === "challenge";

    const fields = isChallenge
      ? [
          h("label", { for: `${id}-category` }, "What is the problem?"),
          h(
            "select",
            { id: `${id}-category` },
            h("option", { value: "unsupported" }, "It is not supported by my text"),
            h("option", { value: "incorrect" }, "It is incorrect"),
            h("option", { value: "unfair" }, "It seems unfair or biased"),
            h("option", { value: "other" }, "Something else")
          ),
          h("label", { for: `${id}-reason` }, "Tell us what looks wrong (required)"),
          h("textarea", { id: `${id}-reason`, rows: "3", maxlength: "800", "aria-required": "true" })
        ]
      : [
          h("label", { for: `${id}-reason` }, "What was not helpful? (optional)"),
          h("textarea", { id: `${id}-reason`, rows: "3", maxlength: "500" })
        ];

    const formEl = h(
      "form",
      { class: "coach-inline-form", novalidate: true },
      h("p", { class: "coach-note" }, "Do not include personal details. Only your words here are saved, not your CV text."),
      fields,
      h(
        "div",
        { class: "coach-inline-actions" },
        h("button", { type: "submit", class: "coach-btn" }, isChallenge ? "Send challenge" : "Send rating"),
        h("button", { type: "button", class: "coach-btn coach-btn-quiet", "data-coach-action": "cancel-inline" }, "Cancel")
      )
    );

    formEl.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeInlineForm(node, trigger);
    });
    formEl.addEventListener("submit", async (event) => {
      event.preventDefault();
      const reason = formEl.querySelector("textarea").value.trim();
      if (isChallenge && reason.length < 5) {
        inlineStatus(node, "Error: please describe the problem in at least a few words.");
        formEl.querySelector("textarea").focus();
        return;
      }
      const submit = formEl.querySelector('button[type="submit"]');
      submit.disabled = true;
      inlineStatus(node, "Working: sending…");
      try {
        if (isChallenge) {
          const category = formEl.querySelector("select").value;
          const result = await api("/coach/challenge", {
            method: "POST",
            body: JSON.stringify({ response_id: resp.response_id, category, reason, focus: resp.focus })
          });
          inlineStatus(node, `Done: ${result.message}`);
        } else {
          await api("/coach/feedback", {
            method: "POST",
            body: JSON.stringify({ response_id: resp.response_id, rating: "down", reason, focus: resp.focus })
          });
          markRated(node, "down");
          inlineStatus(node, "Done: thanks. Your rating was saved anonymously.");
        }
        closeInlineForm(node, trigger);
      } catch (error) {
        submit.disabled = false;
        inlineStatus(node, `Error: ${error.message}`);
      }
    });

    node.querySelector(".coach-inline-slot").append(formEl);
    formEl.querySelector("textarea, select").focus();
  }

  function markRated(node, rating) {
    node.querySelectorAll('[data-coach-action="up"], [data-coach-action="down"]').forEach((button) => {
      button.disabled = true;
      button.setAttribute("aria-pressed", button.dataset.coachAction === rating ? "true" : "false");
    });
  }

  async function rateUp(node) {
    const { resp } = messageData.get(node);
    inlineStatus(node, "Working: sending…");
    try {
      await api("/coach/feedback", {
        method: "POST",
        body: JSON.stringify({ response_id: resp.response_id, rating: "up", focus: resp.focus })
      });
      markRated(node, "up");
      inlineStatus(node, "Done: thanks. Your rating was saved anonymously.");
    } catch (error) {
      inlineStatus(node, `Error: ${error.message}`);
    }
  }

  /* ---------- onboarding: system does it, co-pilot, user does it ---------- */

  function renderStage() {
    steps.forEach((li) => {
      const n = Number(li.dataset.step);
      li.classList.toggle("is-current", n === state.stage);
      li.classList.toggle("is-done", n < state.stage);
      if (n === state.stage) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
      const label = li.querySelector(".coach-step-state");
      if (label) label.textContent = n < state.stage ? "Done" : n === state.stage ? "Current step" : "Next";
    });
    stepStatus.textContent = STAGE_TEXT[state.stage];
    collapseOnboarding(state.stage === 3);
  }

  function collapseOnboarding(collapsed) {
    onboardingBody.hidden = collapsed;
    onboardingToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    onboardingToggle.textContent = collapsed ? "Show quick start" : "Hide quick start";
  }

  function rememberOnboarding() {
    try { localStorage.setItem(ONBOARDING_KEY, "1"); } catch (_) { /* storage may be blocked */ }
  }

  function advanceStage(options) {
    if (state.stage === 1 && options.demo) {
      state.stage = 2;
    } else if (!options.demo && !options.rerunLabel && state.stage < 3) {
      const wasGuided = state.stage === 2;
      state.stage = 3;
      rememberOnboarding();
      appendGuide(
        (wasGuided ? "Nice work. " : "") +
        "From now on you work on your own. Help stays here: “Why this?” under every reply, the focus and plain-language controls, and “Show quick start” if you want the steps again."
      );
    } else {
      return;
    }
    renderStage();
  }

  function applyPersona(id, quiet) {
    const persona = state.greeting && state.greeting.personas.find((p) => p.id === id);
    if (!persona) return;
    state.persona = persona;
    plainBox.checked = persona.plain_language;
    alwaysWhyBox.checked = persona.always_show_evidence;
    setFocusValue(persona.focus);
    if (!quiet) {
      setStatus("ready", `Settings adjusted: focus ${FOCUS_LABEL[persona.focus]}, plain language ${persona.plain_language ? "on" : "off"}, evidence ${persona.always_show_evidence ? "shown automatically" : "shown on request"}. You can change them any time.`);
    }
  }

  function renderPersonaOptions(personas) {
    personaBox.replaceChildren(
      ...personas.map((p, index) =>
        h(
          "label",
          { class: "coach-radio" },
          h("input", { type: "radio", name: "coach-persona", value: p.id, checked: index === 0 }),
          h("span", null, p.label)
        )
      )
    );
  }

  async function runDemo() {
    if (!state.persona) { setStatus("error", "The coach is still loading. Please wait a moment and try again."); return; }
    const sample = state.persona.sample;
    appendGuide("Here is an example. I work on a fictional text, then you can open “Why this?” to see the evidence and the rubric behind my feedback.");
    appendUser(sample.message, `Example input (fictional): ${sample.title}`);
    await send(
      {
        message: sample.message,
        job_description: sample.job_description,
        interview_question: sample.interview_question,
        focus: state.persona.focus,
        plain_language: state.persona.plain_language,
        language: "auto",
        history: []
      },
      { demo: true, forceWhy: true }
    );
  }

  function fillTemplate() {
    if (!state.persona) { setStatus("error", "The coach is still loading. Please wait a moment and try again."); return; }
    const template = state.persona.sample.template;
    input.value = template;
    updateCount();
    hintEl.textContent = "Replace the text in [brackets] with your own, then press Send.";
    appendGuide("Your turn, with my help: 1) Replace the text in [brackets] with your own fictional text. 2) Choose a feedback focus if you want. 3) Press Send, then open “Why this?”.");
    state.stage = Math.max(state.stage, 2);
    renderStage();
    input.focus();
    const start = template.indexOf("[");
    const end = template.indexOf("]", start);
    if (start !== -1 && end !== -1) input.setSelectionRange(start, end + 1);
  }

  /* ---------- loading ---------- */

  async function loadGreeting() {
    if (state.greeting) return;
    try {
      state.greeting = await api("/coach/greeting?language=en", { method: "GET" });
      renderGreeting(state.greeting);
      renderPersonaOptions(state.greeting.personas);
      applyPersona(state.greeting.personas[0].id, true);
      setStatus("ready", "Type a message below, or start with the quick start.");
    } catch (error) {
      appendNotice("error", error.message, loadGreeting);
      setStatus("error", error.message);
    }
  }

  function restoreOnboarding() {
    let done = false;
    try { done = localStorage.getItem(ONBOARDING_KEY) === "1"; } catch (_) { /* ignore */ }
    if (done) state.stage = 3;
    renderStage();
  }

  function start() {
    if (state.started) return;
    state.started = true;
    restoreOnboarding();
    loadGreeting();
  }

  /* ---------- events ---------- */

  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) {
      input.setAttribute("aria-invalid", "true");
      setStatus("error", "Please type or paste some text first.");
      input.focus();
      return;
    }
    input.removeAttribute("aria-invalid");
    const request = readRequest(message);
    appendUser(message);
    send(request, { clearInput: true });
  });

  focusFieldset.addEventListener("change", () => {
    const focus = selectedFocus();
    if (!state.lastRequest) {
      setStatus("ready", `Focus set to ${FOCUS_LABEL[focus]}. It applies to your next message.`);
      return;
    }
    rerun({ focus }, `Focus changed to ${FOCUS_LABEL[focus]}`);
  });

  plainBox.addEventListener("change", () => {
    setStatus("ready", `Plain-language mode ${plainBox.checked ? "on" : "off"}. It applies to your next message.`);
  });

  alwaysWhyBox.addEventListener("change", () => {
    setStatus("ready", `Evidence and rubric will be ${alwaysWhyBox.checked ? "shown automatically" : "shown when you choose Why this?"}.`);
  });

  personaBox.addEventListener("change", (event) => {
    if (event.target.name === "coach-persona") applyPersona(event.target.value);
  });

  section.addEventListener("click", (event) => {
    const button = event.target.closest("[data-coach-action]");
    if (!button || !section.contains(button)) return;
    const action = button.dataset.coachAction;
    const node = button.closest(".coach-msg-bot");

    switch (action) {
      case "why": toggleWhy(node); break;
      case "simpler":
        plainBox.checked = true;
        rerun({ plain_language: true }, "Simpler explanation");
        break;
      case "up": rateUp(node); break;
      case "down":
      case "challenge": openInlineForm(node, action, button); break;
      case "cancel-inline": closeInlineForm(node, node.querySelector('[data-coach-action="challenge"]')); break;
      case "onboarding-toggle": collapseOnboarding(onboardingBody.hidden ? false : true); break;
      case "onboarding-demo": runDemo(); break;
      case "onboarding-template": fillTemplate(); break;
      case "clear":
        log.replaceChildren();
        state.history = [];
        state.lastRequest = null;
        if (state.greeting) renderGreeting(state.greeting);
        setStatus("ready", "Conversation cleared. Type a new message.");
        break;
      default: break;
    }
  });

  /* The shell shows views by toggling .active-view, so start when the coach view opens. */
  const title = $("#coach-title");
  new MutationObserver(() => {
    if (section.classList.contains("active-view")) {
      start();
      if (title) title.focus({ preventScroll: true });
    }
  }).observe(section, { attributes: true, attributeFilter: ["class"] });
  if (section.classList.contains("active-view")) start();

  updateCount();
})();
