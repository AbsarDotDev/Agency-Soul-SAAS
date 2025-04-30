// Script Blocker
(function() {
    // Method 1: Block using MutationObserver to prevent dynamic script injection
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.tagName === 'SCRIPT' && node.src && node.src.includes('envato.workdo.io/verify.js')) {
                    node.remove();
                }
            });
        });
    });

    observer.observe(document, {
        childList: true,
        subtree: true
    });

    // Method 2: Override appendChild and insertBefore to prevent script injection
    const originalAppendChild = Element.prototype.appendChild;
    const originalInsertBefore = Element.prototype.insertBefore;

    Element.prototype.appendChild = function(node) {
        if (node.tagName === 'SCRIPT' && node.src && node.src.includes('envato.workdo.io/verify.js')) {
            return node; // Block the script
        }
        return originalAppendChild.call(this, node);
    };

    Element.prototype.insertBefore = function(node, referenceNode) {
        if (node.tagName === 'SCRIPT' && node.src && node.src.includes('envato.workdo.io/verify.js')) {
            return node; // Block the script
        }
        return originalInsertBefore.call(this, node, referenceNode);
    };

    // Method 3: Block using Service Worker if supported
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw-blocker.js').catch(function(error) {
            console.log('Service Worker registration failed:', error);
        });
    }

    // Method 4: Override fetch and XMLHttpRequest
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        if (typeof url === 'string' && url.includes('envato.workdo.io/verify.js')) {
            return new Promise(function(resolve) {
                resolve(new Response('', {
                    status: 200,
                    headers: { 'Content-Type': 'application/javascript' }
                }));
            });
        }
        return originalFetch.apply(this, arguments);
    };

    const originalXHROpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {
        if (typeof url === 'string' && url.includes('envato.workdo.io/verify.js')) {
            url = 'data:application/javascript;base64,'; // Return empty script
        }
        return originalXHROpen.apply(this, arguments);
    };

    // Method 5: Define a blocked domains list and check all script sources
    const blockedDomains = ['envato.workdo.io'];
    
    // Override createElement to catch script creation
    const originalCreateElement = document.createElement;
    document.createElement = function(tagName) {
        const element = originalCreateElement.call(document, tagName);
        if (tagName.toLowerCase() === 'script') {
            const originalSetAttribute = element.setAttribute;
            element.setAttribute = function(name, value) {
                if (name === 'src' && blockedDomains.some(domain => value.includes(domain))) {
                    return;
                }
                return originalSetAttribute.call(this, name, value);
            };
        }
        return element;
    };
})(); 