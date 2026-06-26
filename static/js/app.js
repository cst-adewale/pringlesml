// app.js — Shared JavaScript for Delivery Prediction System

// Health check on page load
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.status !== 'ok') console.warn('API health check failed');
    } catch (e) {
        console.warn('API not reachable:', e.message);
    }
}

checkHealth();
