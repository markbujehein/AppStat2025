// Landing page: filter and regroup the demos in data (embedded JSON), nothing is fetched.
const demos = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = { q: params.get("find") ?? "", topic: params.get("topic") ?? "", group: params.get("group") ?? "week" };
const haystack = demos.map((d) => [d.title, d.description, ...d.categories, ...d.headings, ...d.takeaways].join(" ").toLowerCase());
const terms = () => state.q.toLowerCase().split(/\s+/).filter(Boolean);
const matches = (i, topic) => terms().every((t) => haystack[i].includes(t)) && (!topic || demos[i].categories.includes(topic));

function marked(node, text, found) {
  if (!found.length) return node.append(text);
  const re = new RegExp("(" + found.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "gi");
  text.split(re).forEach((part, i) => {
    if (i % 2) node.append(Object.assign(document.createElement("mark"), { textContent: part }));
    else node.append(part);
  });
}

function card(d) {
  const found = terms();
  const el = document.createElement("article");
  el.className = "demo";
  const a = Object.assign(document.createElement("a"), { href: d.path });
  marked(a, d.title, found);
  const h = document.createElement("h3");
  h.append(a);
  const p = document.createElement("p");
  marked(p, d.description, found);
  el.append(h, p);
  const hits = found.length ? d.takeaways.filter((t) => found.some((w) => t.toLowerCase().includes(w))) : [];
  if (d.takeaways.length) {
    const details = document.createElement("details");
    details.open = hits.length > 0;
    const sum = document.createElement("summary");
    sum.textContent = hits.length ? `${hits.length} matching key points` : `Key points (${d.takeaways.length})`;
    const ul = document.createElement("ul");
    for (const t of hits.length ? hits : d.takeaways) {
      const li = document.createElement("li");
      marked(li, t, found);
      ul.append(li);
    }
    details.append(sum, ul);
    el.append(details);
  }
  const tags = document.createElement("div");
  tags.className = "tags";
  for (const topic of d.categories) {
    const b = Object.assign(document.createElement("button"), { type: "button", textContent: topic });
    b.onclick = () => set({ topic });
    tags.append(b);
  }
  el.append(tags);
  return el;
}

function group(label, big, items) {
  const s = document.createElement("section");
  s.className = "group";
  const h = document.createElement("h2");
  h.append(big);
  if (label) h.append(Object.assign(document.createElement("small"), { textContent: label }));
  s.append(h, Object.assign(document.createElement("div"), {}));
  s.lastChild.append(...items.map(card));
  return s;
}

function render() {
  const topics = [...new Set(demos.flatMap((d) => d.categories))].sort((a, b) => a.localeCompare(b));
  const rail = $("topics");
  rail.replaceChildren(
    ...["", ...topics].map((topic) => {
      const n = demos.filter((_, i) => matches(i, topic)).length;
      const li = document.createElement("li");
      const b = Object.assign(document.createElement("button"), { type: "button" });
      b.setAttribute("aria-pressed", String(state.topic === topic));
      b.append(topic || "All", Object.assign(document.createElement("span"), { textContent: n }));
      b.onclick = () => set({ topic });
      li.append(b);
      return li;
    }),
  );
  document.querySelectorAll("[data-group]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.group === state.group)));
  const shown = demos.filter((_, i) => matches(i, state.topic));
  const sections = [];
  if (state.group === "week") {
    for (const w of [...new Set(shown.map((d) => d.week))].sort((a, b) => a - b))
      sections.push(group("Week", String(w), shown.filter((d) => d.week === w)));
  } else {
    for (const t of topics) {
      const inTopic = shown.filter((d) => d.categories.includes(t));
      if (inTopic.length) sections.push(group("", t, inTopic));
    }
  }
  $("list").replaceChildren(...(sections.length ? sections : [Object.assign(document.createElement("p"), { className: "empty", textContent: "No demo matches. Clear the search or pick All." })]));
  const next = new URLSearchParams();
  if (state.q) next.set("find", state.q);
  if (state.topic) next.set("topic", state.topic);
  if (state.group !== "week") next.set("group", state.group);
  history.replaceState(null, "", next.size ? "?" + next : location.pathname);
}

function set(change) {
  Object.assign(state, change);
  $("q").value = state.q;
  render();
}

$("q").value = state.q;
$("q").addEventListener("input", (e) => set({ q: e.target.value }));
document.querySelectorAll("[data-group]").forEach((b) => (b.onclick = () => set({ group: b.dataset.group })));
addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); }
  if (e.key === "Escape" && state.q) set({ q: "" });
});
render();
