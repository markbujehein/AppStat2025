// Demo page: read the static export first, start the interactive one only on request.
const $$ = (s) => [...document.querySelectorAll(s)];
const hasStatic = !!document.querySelector('.frame[data-mode="static"]:not([hidden])');

function load(frame) {
  const iframe = frame.querySelector("iframe");
  if (iframe.src) return;
  const skeleton = frame.querySelector(".skeleton");
  skeleton.hidden = false;
  frame.querySelector(".load")?.setAttribute("hidden", "");
  iframe.addEventListener("load", () => {
    skeleton.hidden = true;
    iframe.hidden = false;
    // marimo's page scrolls itself to the end while it renders. Start at the top.
    const top = () => { const app = iframe.contentDocument?.getElementById("App"); if (app) app.scrollTop = 0; };
    [200, 800, 2000].forEach((ms) => setTimeout(top, ms));
  }, { once: true });
  iframe.src = iframe.dataset.src;
}

function show(mode) {
  $$(".frame").forEach((f) => (f.hidden = f.dataset.mode !== mode));
  $$(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  history.replaceState(null, "", mode === "app" ? "#interactive" : "#read");
  if (mode === "static") load(document.querySelector('.frame[data-mode="static"]'));
}

$$(".modes button").forEach((b) => (b.onclick = () => show(b.dataset.mode)));
document.querySelector(".run").onclick = () => load(document.querySelector('.frame[data-mode="app"]'));
show(!hasStatic || location.hash === "#interactive" ? "app" : "static");
if (location.hash === "#interactive") document.querySelector(".run").click();
navigator.serviceWorker?.register("../../sw.js");
