// ── Emotion Mirror Dashboard ─────────────────────────────────────────

// ── Config ───────────────────────────────────────────────────────────
const VISION_HOST = location.hostname + ':8630';
const VISION_API = `http://${VISION_HOST}`;
const DASHBOARD_WS = `ws://${location.host}/ws`;

const EMOTION_COLORS = {
    angry: '#e94560', happy: '#2ecc71', neutral: '#3498db',
    sad: '#2c3e50', surprised: '#f39c12', fear: '#9b59b6',
    thinking: '#f39c12', confused: '#a0a0c0', curious: '#00d2ff',
    excited: '#2ecc71', laugh: '#2ecc71', listening: '#3498db',
    Anger: '#e94560', Happiness: '#2ecc71', Neutral: '#3498db',
    Sadness: '#2c3e50', Surprise: '#f39c12', Fear: '#9b59b6',
    Contempt: '#a0a0c0', Disgust: '#8b5e3c',
};

// ── State ────────────────────────────────────────────────────────────
let visionWs = null;
let dashboardWs = null;
let latestFaces = null;
let currentLlmText = '';
let currentRunId = null;
let logEntries = [];
let currentMode = 'conversation';
let uploadFiles = [];

// ── DOM refs ─────────────────────────────────────────────────────────
const videoEl = document.getElementById('video-stream');
const canvasEl = document.getElementById('overlay-canvas');
const ctx = canvasEl.getContext('2d');
const noVideoEl = document.getElementById('no-video');
const asrTextEl = document.getElementById('asr-text');
const logEntriesEl = document.getElementById('log-entries');

// ── Toast ────────────────────────────────────────────────────────────
function showToast(msg, isError = false) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = isError ? 'toast show error' : 'toast show';
    setTimeout(() => { el.className = 'toast'; }, 2500);
}

// ── Vision WebSocket (face detection) ────────────────────────────────
let visionRetry = 1000;

function connectVision() {
    const url = `ws://${VISION_HOST}/ws`;
    visionWs = new WebSocket(url);

    visionWs.onopen = () => {
        console.log('Vision WS connected');
        visionRetry = 1000;
        document.getElementById('dot-vision').className = 'dot live';
    };

    visionWs.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            latestFaces = data.faces || [];
        } catch(err) {
            console.error('Vision WS parse error:', err);
        }
    };

    visionWs.onclose = () => {
        document.getElementById('dot-vision').className = 'dot off';
        latestFaces = null;
        visionRetry = Math.min(visionRetry * 1.5, 10000);
        setTimeout(connectVision, visionRetry);
    };

    visionWs.onerror = () => visionWs.close();
}

// ── Dashboard WebSocket (ASR, LLM, state) ────────────────────────────
let dashRetry = 1000;

function connectDashboard() {
    dashboardWs = new WebSocket(DASHBOARD_WS);

    dashboardWs.onopen = () => {
        console.log('Dashboard WS connected');
        dashRetry = 1000;
        document.getElementById('dot-jetson').className = 'dot live';
    };

    dashboardWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            handleDashboardMsg(msg);
        } catch(err) {
            console.error('Dashboard WS parse error:', err);
        }
    };

    dashboardWs.onclose = () => {
        document.getElementById('dot-jetson').className = 'dot off';
        dashRetry = Math.min(dashRetry * 1.5, 10000);
        setTimeout(connectDashboard, dashRetry);
    };

    dashboardWs.onerror = () => dashboardWs.close();
}

function handleDashboardMsg(msg) {
    switch(msg.type) {
        case 'asr_partial':
            asrTextEl.textContent = msg.text;
            asrTextEl.className = 'asr-text partial';
            break;

        case 'asr_final':
            asrTextEl.textContent = msg.text;
            asrTextEl.className = 'asr-text';
            break;

        case 'llm_delta':
            if (msg.run_id !== currentRunId) {
                currentRunId = msg.run_id;
                currentLlmText = '';
                addLogEntry();
            }
            currentLlmText += msg.text;
            updateCurrentLog();
            break;

        case 'llm_end':
            if (msg.run_id === currentRunId) {
                currentLlmText = msg.full_text;
                updateCurrentLog(true);
                currentRunId = null;
            }
            break;

        case 'state':
            updateState(msg.state);
            break;

        case 'emotion':
            break;

        case 'robot_state':
            updateRobotState(msg);
            break;

        case 'mode_changed':
            currentMode = msg.mode;
            syncModeUI();
            showToast('Mode: ' + msg.mode);
            break;
    }
}

// ── Observation Log ──────────────────────────────────────────────────
function addLogEntry() {
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8);
    logEntries.push({ timestamp: ts, text: '', done: false });
    renderLog();
}

function updateCurrentLog(done = false) {
    if (logEntries.length === 0) addLogEntry();
    const last = logEntries[logEntries.length - 1];
    last.text = currentLlmText;
    last.done = done;
    renderLog();
}

function renderLog() {
    if (logEntries.length === 0) {
        logEntriesEl.innerHTML = '<div class="log-entry" style="color: var(--text-dim)">Waiting for conversation...</div>';
        return;
    }

    if (logEntries.length > 20) logEntries = logEntries.slice(-20);

    let html = '';
    for (let i = 0; i < logEntries.length; i++) {
        const entry = logEntries[i];
        const isCurrent = i === logEntries.length - 1 && !entry.done;
        const cls = isCurrent ? 'log-entry current' : 'log-entry';
        const cursor = isCurrent ? '<span class="typing-cursor"></span>' : '';
        html += `<div class="${cls}">
            <div class="timestamp">${entry.timestamp}</div>
            <div>${entry.text}${cursor}</div>
        </div>`;
    }
    logEntriesEl.innerHTML = html;

    const container = document.getElementById('observation-log');
    container.scrollTop = container.scrollHeight;
}

// ── State & Robot State ──────────────────────────────────────────────
function updateState(state) {
    const el = document.getElementById('robot-state');
    el.textContent = state;
    el.dataset.state = state;
}

function updateRobotState(msg) {
    // Mode
    if (msg.mode) {
        currentMode = msg.mode;
        document.getElementById('robot-mode').textContent = msg.mode;
        syncModeUI();
    }

    // Emotion
    const emotionEl = document.getElementById('robot-emotion');
    const color = EMOTION_COLORS[msg.emotion] || '#3498db';
    emotionEl.innerHTML = `<span class="emotion-tag" style="color:${color}; border-color:${color}40; background:${color}15">${msg.emotion}</span>`;

    // Head
    document.getElementById('robot-head').textContent =
        `Y:${msg.head.yaw.toFixed(1)} P:${msg.head.pitch.toFixed(1)} R:${msg.head.roll.toFixed(1)}`;

    // Antenna
    document.getElementById('robot-antenna').textContent =
        `L:${msg.antenna.left.toFixed(1)} R:${msg.antenna.right.toFixed(1)}`;

    // Tracking
    const trackEl = document.getElementById('robot-tracking');
    trackEl.textContent = `${msg.tracking.source} (${(msg.tracking.confidence * 100).toFixed(0)}%)`;
    trackEl.className = msg.tracking.source === 'face' ? 'value highlight' : 'value';

    // Speaking
    const speakEl = document.getElementById('robot-speaking');
    speakEl.textContent = msg.speaking ? 'yes' : 'no';
    speakEl.className = msg.speaking ? 'value highlight' : 'value';

    // Robot connected indicator
    document.getElementById('dot-robot').className = 'dot live';

    // Emotion mapping
    const mappingEl = document.getElementById('emotion-mapping');
    const descEl = document.getElementById('mapping-desc');
    if (msg.emotion_mapping) {
        mappingEl.style.display = 'block';
        const m = msg.emotion_mapping;
        descEl.textContent = `${m.description} | Antenna: L${m.antenna_target.left} R${m.antenna_target.right}`;
    } else {
        mappingEl.style.display = 'none';
    }
}

// ── Canvas overlay (face detection) ──────────────────────────────────
function drawOverlay() {
    const rect = videoEl.getBoundingClientRect();
    canvasEl.width = rect.width;
    canvasEl.height = rect.height;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!latestFaces || latestFaces.length === 0) {
        requestAnimationFrame(drawOverlay);
        return;
    }

    const cw = canvasEl.width;
    const ch = canvasEl.height;

    for (const face of latestFaces) {
        const [x1, y1, x2, y2] = face.bbox;
        const color = EMOTION_COLORS[face.emotion] || '#3498db';

        const bx = x1 * cw;
        const by = y1 * ch;
        const bw = (x2 - x1) * cw;
        const bh = (y2 - y1) * ch;

        const cornerLen = Math.min(bw, bh) * 0.2;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.8;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(bx, by + cornerLen);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + cornerLen, by);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + cornerLen);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - cornerLen);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + cornerLen, by + bh);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - cornerLen);
        ctx.stroke();

        ctx.globalAlpha = 1.0;

        // Labels
        ctx.font = '12px monospace';
        ctx.fillStyle = color;

        const identity = face.identity || '?';
        const conf = ((face.emotion_confidence || 0) * 100).toFixed(0);
        let labelY = by - 6;
        if (labelY < 14) labelY = by + bh + 16;

        const labelText = `${identity} | ${face.emotion} ${conf}%`;
        const metrics = ctx.measureText(labelText);
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(bx - 1, labelY - 12, metrics.width + 6, 16);
        ctx.fillStyle = color;
        ctx.fillText(labelText, bx + 2, labelY);

        // 5-point landmarks
        if (face.landmarks) {
            ctx.fillStyle = '#00d2ff';
            for (const [lx, ly] of face.landmarks) {
                ctx.beginPath();
                ctx.arc(lx * cw, ly * ch, 2.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }

    requestAnimationFrame(drawOverlay);
}

// ── MJPEG video ──────────────────────────────────────────────────────
function setupVideo() {
    const streamUrl = `http://${VISION_HOST}/stream`;
    videoEl.src = streamUrl;

    videoEl.onload = () => {
        videoEl.style.display = 'block';
        noVideoEl.style.display = 'none';
    };

    videoEl.onerror = () => {
        videoEl.style.display = 'none';
        noVideoEl.style.display = 'flex';
        setTimeout(() => { videoEl.src = streamUrl; }, 5000);
    };
}

// ── Settings Panel ───────────────────────────────────────────────────
function initSettings() {
    const overlay = document.getElementById('settings-overlay');
    document.getElementById('settings-open').onclick = () => {
        overlay.classList.add('open');
        loadFaces();
    };
    document.getElementById('settings-close').onclick = () => {
        overlay.classList.remove('open');
    };
    overlay.onclick = (e) => {
        if (e.target === overlay) overlay.classList.remove('open');
    };

    // Tabs
    document.querySelectorAll('.settings-tab').forEach(tab => {
        tab.onclick = () => {
            document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
        };
    });

    // Mode selection
    document.querySelectorAll('.mode-option').forEach(opt => {
        opt.onclick = () => {
            const mode = opt.dataset.mode;
            if (mode === currentMode) return;
            setMode(mode);
        };
    });

    // Live enroll
    document.getElementById('enroll-live-btn').onclick = enrollLive;

    // File upload
    const uploadArea = document.getElementById('upload-area');
    const uploadInput = document.getElementById('upload-input');

    uploadArea.onclick = () => uploadInput.click();
    uploadInput.onchange = () => handleUploadFiles(uploadInput.files);

    uploadArea.ondragover = (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); };
    uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
    uploadArea.ondrop = (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleUploadFiles(e.dataTransfer.files);
    };

    document.getElementById('upload-btn').onclick = uploadAndEnroll;

    // Export / Import
    document.getElementById('export-btn').onclick = exportFaces;
    document.getElementById('import-btn').onclick = () => document.getElementById('import-input').click();
    document.getElementById('import-input').onchange = importFaces;
}

function syncModeUI() {
    document.querySelectorAll('.mode-option').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.mode === currentMode);
        opt.querySelector('input').checked = opt.dataset.mode === currentMode;
    });
    document.getElementById('mode-status').textContent = 'Current: ' + currentMode;
}

function setMode(mode) {
    if (!dashboardWs || dashboardWs.readyState !== 1) {
        showToast('Not connected', true);
        return;
    }
    dashboardWs.send(JSON.stringify({ type: 'set_mode', mode }));
}

// ── Face Management API ──────────────────────────────────────────────
async function loadFaces() {
    try {
        const res = await fetch(`${VISION_API}/api/faces`);
        const data = await res.json();
        renderFaceList(data.faces || []);
    } catch (e) {
        renderFaceList([]);
    }
}

function renderFaceList(faces) {
    const el = document.getElementById('face-list');
    if (!faces.length) {
        el.innerHTML = '<div class="face-empty">No faces registered</div>';
        return;
    }
    el.innerHTML = faces.map(name =>
        `<div class="face-item">
            <span class="face-name">${name}</span>
            <button class="face-item-btn" onclick="deleteFace('${name}')">Delete</button>
        </div>`
    ).join('');
}

async function deleteFace(name) {
    try {
        await fetch(`${VISION_API}/api/faces/${encodeURIComponent(name)}`, { method: 'DELETE' });
        showToast(`Deleted: ${name}`);
        loadFaces();
    } catch (e) {
        showToast('Delete failed', true);
    }
}

async function enrollLive() {
    const name = document.getElementById('enroll-name').value.trim();
    if (!name) { showToast('Enter a name', true); return; }

    const btn = document.getElementById('enroll-live-btn');
    btn.disabled = true;
    try {
        const res = await fetch(`${VISION_API}/api/faces/enroll?name=${encodeURIComponent(name)}`, { method: 'POST' });
        const data = await res.json();
        if (data.error) { showToast(data.error, true); return; }
        showToast(`Enrolled: ${name}`);
        document.getElementById('enroll-name').value = '';
        loadFaces();
    } catch (e) {
        showToast('Enroll failed', true);
    } finally {
        btn.disabled = false;
    }
}

function handleUploadFiles(files) {
    uploadFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    const preview = document.getElementById('upload-preview');
    preview.innerHTML = '';
    uploadFiles.forEach(f => {
        const img = document.createElement('img');
        img.className = 'upload-thumb';
        img.src = URL.createObjectURL(f);
        preview.appendChild(img);
    });
    document.getElementById('upload-btn').disabled = uploadFiles.length === 0;
}

async function uploadAndEnroll() {
    const name = document.getElementById('upload-name').value.trim();
    if (!name) { showToast('Enter a name', true); return; }
    if (!uploadFiles.length) return;

    const btn = document.getElementById('upload-btn');
    btn.disabled = true;
    let ok = 0, fail = 0;

    for (const file of uploadFiles) {
        const fd = new FormData();
        fd.append('name', name);
        fd.append('image', file);
        try {
            const res = await fetch(`${VISION_API}/api/faces/enroll-image`, { method: 'POST', body: fd });
            const data = await res.json();
            if (data.error) { fail++; } else { ok++; }
        } catch (e) {
            fail++;
        }
    }

    showToast(`Enrolled ${ok}/${uploadFiles.length} images` + (fail ? ` (${fail} failed)` : ''));
    uploadFiles = [];
    document.getElementById('upload-preview').innerHTML = '';
    document.getElementById('upload-input').value = '';
    btn.disabled = true;
    loadFaces();
}

async function exportFaces() {
    try {
        const res = await fetch(`${VISION_API}/api/faces/export`);
        if (!res.ok) { showToast('Export failed', true); return; }
        const blob = await res.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'faces.zip';
        a.click();
        URL.revokeObjectURL(a.href);
        showToast('Exported faces.zip');
    } catch (e) {
        showToast('Export failed', true);
    }
}

async function importFaces() {
    const input = document.getElementById('import-input');
    if (!input.files.length) return;

    const fd = new FormData();
    fd.append('file', input.files[0]);

    try {
        const res = await fetch(`${VISION_API}/api/faces/import`, { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) { showToast(data.error, true); return; }
        showToast(`Imported ${(data.faces || []).length} faces`);
        loadFaces();
    } catch (e) {
        showToast('Import failed', true);
    }
    input.value = '';
}

// ── Init ─────────────────────────────────────────────────────────────
function init() {
    setupVideo();
    connectVision();
    connectDashboard();
    initSettings();
    requestAnimationFrame(drawOverlay);
}

init();
