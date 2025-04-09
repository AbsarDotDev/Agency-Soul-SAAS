// Service Worker to block verify.js
self.addEventListener('install', function(event) {
    self.skipWaiting();
});

self.addEventListener('activate', function(event) {
    event.waitUntil(clients.claim());
});

self.addEventListener('fetch', function(event) {
    const url = event.request.url;
    
    // Block verify.js requests
    if (url.includes('envato.workdo.io/verify.js')) {
        event.respondWith(
            new Response('', {
                status: 200,
                headers: { 'Content-Type': 'application/javascript' }
            })
        );
        return;
    }
    
    // Let other requests pass through
    event.respondWith(fetch(event.request));
}); 