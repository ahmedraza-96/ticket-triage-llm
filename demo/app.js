/* Ticket Triage Console — streams /v1/chat/completions from the vLLM endpoint
   (set in config.js) and renders the JSON triage live: the category badge
   appears the moment the "category" field closes in the stream; the reply
   streams in with {{Placeholder}} slots rendered as highlighted chips. */

const CATEGORY_HUES = {
  ACCOUNT: "#7dd3fc", CANCEL: "#f87171", CONTACT: "#a5b4fc", DELIVERY: "#4ade80",
  FEEDBACK: "#f0abfc", INVOICE: "#fbbf24", ORDER: "#fb923c", PAYMENT: "#34d399",
  REFUND: "#f97316", SHIPPING: "#38bdf8", SUBSCRIPTION: "#c084fc",
};

const SYSTEM_PROMPT = window.TRIAGE_SYSTEM_PROMPT; // set by config.js

const SAMPLES = [
  { label: "refund", text: "I bought a blender two weeks ago and it just stopped working mid-smoothie. I want my money back, how do I do that?" },
  { label: "delivery", text: "My package was supposed to arrive on Monday and it's Thursday now. Where is my order??" },
  { label: "cancel", text: "I've decided I don't need the standing desk anymore, please cancel my purchase before it ships." },
  { label: "account", text: "I can't log into my profile anymore, I think my account got locked after too many password attempts." },
  { label: "invoice", text: "I need a copy of the invoice for my last order for my company's expense report." },
];

const $ = (id) => document.getElementById(id);

function init() {
  const wrap = $("samples");
  for (const s of SAMPLES) {
    const b = document.createElement("button");
    b.textContent = s.label;
    b.onclick = () => { $("ticket").value = s.text; };
    wrap.appendChild(b);
  }
  $("triage").onclick = triage;
  $("show-raw").onchange = (e) => $("raw").classList.toggle("hidden", !e.target.checked);
  checkEndpoint();
}

async function checkEndpoint() {
  const el = $("endpoint-status");
  try {
    const r = await fetch(`${window.VLLM_BASE_URL}/models`);
    const data = await r.json();
    window.MODEL_ID = data.data[0].id;
    el.innerHTML = `endpoint <span class="ok">●</span> ${window.MODEL_ID}`;
  } catch {
    el.innerHTML = `endpoint <span class="bad">●</span> unreachable — set VLLM_BASE_URL in config.js`;
  }
}

function renderReply(text) {
  // {{Placeholder}} slots become highlighted chips — trained-in policy safety.
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/\{\{([^}]+)\}\}/g, '<span class="slot">$1</span>');
}

async function triage() {
  const ticket = $("ticket").value.trim();
  if (!ticket) return;
  const btn = $("triage");
  btn.disabled = true;
  $("category").classList.add("hidden");
  $("reply").innerHTML = "";
  $("raw").textContent = "";
  $("timing").textContent = "";

  const t0 = performance.now();
  let raw = "", firstToken = null, tokens = 0;

  try {
    const resp = await fetch(`${window.VLLM_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer local" },
      body: JSON.stringify({
        model: window.MODEL_ID,
        stream: true,
        temperature: 0,
        max_tokens: 400,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: ticket },
        ],
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line.includes("[DONE]")) continue;
        const delta = JSON.parse(line.slice(6)).choices?.[0]?.delta?.content;
        if (!delta) continue;
        if (firstToken === null) firstToken = performance.now() - t0;
        tokens += 1;
        raw += delta;
        $("raw").textContent = raw;
        update(raw);
      }
    }
    update(raw, true);
    const total = (performance.now() - t0) / 1000;
    $("timing").textContent =
      `TTFT ${(firstToken / 1000).toFixed(2)}s · ${total.toFixed(1)}s total · ~${(tokens / total).toFixed(0)} tok/s`;
  } catch (err) {
    $("reply").innerHTML = `<span class="bad">Request failed: ${err.message}</span>`;
  } finally {
    btn.disabled = false;
  }
}

function update(raw, final = false) {
  // Category: visible as soon as `"category": "..."` closes in the stream.
  const cat = raw.match(/"category"\s*:\s*"([A-Z]+)"/);
  if (cat && CATEGORY_HUES[cat[1]]) {
    const badge = $("category");
    badge.textContent = cat[1];
    badge.style.color = CATEGORY_HUES[cat[1]];
    badge.style.borderColor = CATEGORY_HUES[cat[1]];
    badge.classList.remove("hidden");
  }
  // Reply: stream the partial string value of "reply".
  const rep = raw.match(/"reply"\s*:\s*"((?:[^"\\]|\\.)*)("?)/);
  if (rep) {
    let text = rep[1];
    try { text = JSON.parse(`"${text}"`); } catch { /* mid-escape; render as-is */ }
    $("reply").innerHTML = renderReply(text);
  }
  if (final && !rep) $("reply").textContent = raw; // non-JSON fallback (base model!)
}

init();
