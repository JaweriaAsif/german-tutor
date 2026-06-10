"use strict";

const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const sendBtn = document.getElementById("send");
const learnerInput = document.getElementById("learner");
const summaryEl = document.getElementById("summary");
const player = document.getElementById("player");

learnerInput.value = localStorage.getItem("learner_id") || "default";
const learner = () => (learnerInput.value || "default").trim();

// Escape HTML, then turn [[de:...]] markers into clickable play chips.
function renderTutor(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const parts = [];
  const re = /\[\[de:([\s\S]+?)\]\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(esc(text.slice(last, m.index)));
    const de = m[1].trim();
    const attr = esc(de).replace(/"/g, "&quot;");
    parts.push(
      `<span class="de"><span class="word">${esc(de)}</span>` +
      `<button class="play" title="Hear it" data-text="${attr}">🔊</button></span>`
    );
    last = re.lastIndex;
  }
  parts.push(esc(text.slice(last)));
  return parts.join("");
}

function addMessage(role, text, isHtml = false) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (isHtml) div.innerHTML = text; else div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function playGerman(text, chip) {
  try {
    if (chip) chip.classList.add("playing");
    player.src = "/api/tts?text=" + encodeURIComponent(text) +
      "&voice=" + encodeURIComponent(localStorage.getItem("voice") || "Anna");
    player.onended = () => chip && chip.classList.remove("playing");
    await player.play();
  } catch (e) {
    console.error(e);
    if (chip) chip.classList.remove("playing");
  }
}

chat.addEventListener("click", (e) => {
  const btn = e.target.closest(".play");
  if (btn) playGerman(btn.dataset.text, btn.closest(".de"));
});

async function loadState() {
  try {
    const r = await fetch("/api/state?learner_id=" + encodeURIComponent(learner()));
    const s = await r.json();
    summaryEl.textContent = (s.returning ? "Willkommen zurück! " : "New learner · ") + s.summary;
  } catch (_) { summaryEl.textContent = ""; }
}

async function send(message) {
  if (!message) return;
  addMessage("me", message);
  input.value = "";
  sendBtn.disabled = true;
  const typing = addMessage("sys", "Tutor is typing…");
  typing.classList.add("typing");
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ learner_id: learner(), message }),
    });
    typing.remove();
    if (!r.ok) addMessage("sys", "Error: " + (await r.text()));
    else addMessage("tutor", renderTutor((await r.json()).reply), true);
    loadState();
  } catch (e) {
    typing.remove();
    addMessage("sys", "Network error: " + e.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value.trim()); });
document.querySelector(".quickbar").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-cmd]");
  if (b) send(b.dataset.cmd);
});
learnerInput.addEventListener("change", () => {
  localStorage.setItem("learner_id", learner());
  chat.innerHTML = "";
  loadState();
});

loadState();
addMessage("sys", "Tip: click 🔊 next to any German word to hear it. Try a quick action below.");
