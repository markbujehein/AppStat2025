// Caches what never changes, so a second demo (or a second visit) starts at once.
// demos/assets/ has content-hashed names and the Pyodide URL carries its version: cache first.
// demos/public/ (data, appstat code) can change under the same name: serve the copy, refresh behind it.
const CACHE = "appstat-v1";
const immutable = (url) => url.pathname.includes("/demos/assets/") || url.hostname === "cdn.jsdelivr.net";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(clients.claim()));
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const shared = url.pathname.includes("/demos/public/");
  if (event.request.method !== "GET" || !(shared || immutable(url))) return;
  event.respondWith(
    caches.open(CACHE).then(async (cache) => {
      const cached = await cache.match(event.request);
      const fresh = () => fetch(event.request).then((r) => (r.status === 200 && cache.put(event.request, r.clone()), r));
      if (cached && immutable(url)) return cached;
      return cached ? (fresh().catch(() => {}), cached) : fresh();
    }),
  );
});
