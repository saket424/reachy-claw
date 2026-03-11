/**
 * Vision TRT Frontend — WebSocket client + Canvas overlay rendering.
 */

const EMOTION_COLORS = {
    Anger: "#e94560",
    Contempt: "#a0a0c0",
    Disgust: "#8b5e3c",
    Fear: "#9b59b6",
    Happiness: "#2ecc71",
    Neutral: "#3498db",
    Sadness: "#2c3e50",
    Surprise: "#f39c12",
    angry: "#e94560",
    happy: "#2ecc71",
    neutral: "#3498db",
    sad: "#2c3e50",
    surprised: "#f39c12",
    fear: "#9b59b6",
};

let ws = null;
let latestDetections = null;
let faceDb = [];

// DOM elements
const videoEl = document.getElementById("video-stream");
const canvasEl = document.getElementById("overlay-canvas");
const ctx = canvasEl.getContext("2d");
const fpsEl = document.getElementById("fps-value");
const latencyEl = document.getElementById("latency-value");
const facesEl = document.getElementById("faces-value");
const faceListEl = document.getElementById("face-list");
const detectionInfoEl = document.getElementById("detection-info");
const enrollDialog = document.getElementById("enroll-dialog");
const enrollNameInput = document.getElementById("enroll-name");

// --- WebSocket ---

function connectWS() {
    const wsUrl = `ws://${location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("WebSocket connected");
        document.getElementById("connection-status").style.display = "none";
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            latestDetections = data;
            updateStats(data.stats);
            updateDetectionInfo(data.faces);
        } catch (e) {
            console.error("WS parse error:", e);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket disconnected, reconnecting...");
        document.getElementById("connection-status").style.display = "block";
        setTimeout(connectWS, 2000);
    };

    ws.onerror = () => {
        ws.close();
    };
}

// --- Stats ---

function updateStats(stats) {
    if (!stats) return;
    fpsEl.textContent = stats.fps.toFixed(1);
    latencyEl.textContent = stats.inference_ms.toFixed(1) + " ms";
}

function updateDetectionInfo(faces) {
    if (!faces || faces.length === 0) {
        facesEl.textContent = "0";
        detectionInfoEl.innerHTML = '<span class="label">No faces detected</span>';
        return;
    }

    facesEl.textContent = faces.length;

    let html = "";
    faces.forEach((face, i) => {
        const color = EMOTION_COLORS[face.emotion] || "#3498db";
        const conf = ((face.emotion_confidence || 0) * 100).toFixed(0);
        const identity = face.identity || "Unknown";

        html += `
            <div style="margin-bottom: 8px;">
                <span class="label">Face ${i + 1}:</span> ${identity}<br>
                <div class="emotion-bar">
                    <span style="width: 60px; color: ${color}">${face.emotion}</span>
                    <div class="bar">
                        <div class="bar-fill" style="width: ${conf}%; background: ${color}"></div>
                    </div>
                    <span>${conf}%</span>
                </div>
            </div>
        `;
    });
    detectionInfoEl.innerHTML = html;
}

// --- Canvas overlay ---

function drawOverlay() {
    const rect = videoEl.getBoundingClientRect();
    canvasEl.width = rect.width;
    canvasEl.height = rect.height;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!latestDetections || !latestDetections.faces) {
        requestAnimationFrame(drawOverlay);
        return;
    }

    const cw = canvasEl.width;
    const ch = canvasEl.height;

    for (const face of latestDetections.faces) {
        const [x1, y1, x2, y2] = face.bbox;
        const color = EMOTION_COLORS[face.emotion] || "#3498db";

        // Bbox (normalized coords → pixel)
        const bx = x1 * cw;
        const by = y1 * ch;
        const bw = (x2 - x1) * cw;
        const bh = (y2 - y1) * ch;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);

        // Labels
        ctx.font = "12px monospace";
        ctx.fillStyle = color;

        let labelY = by - 6;
        if (labelY < 14) labelY = by + bh + 16;

        const identity = face.identity || "?";
        const conf = ((face.emotion_confidence || 0) * 100).toFixed(0);
        ctx.fillText(`${identity} | ${face.emotion} ${conf}%`, bx, labelY);

        // 5-point landmarks
        if (face.landmarks) {
            ctx.fillStyle = "#00d2ff";
            for (const [lx, ly] of face.landmarks) {
                ctx.beginPath();
                ctx.arc(lx * cw, ly * ch, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }

    requestAnimationFrame(drawOverlay);
}

// --- Face DB ---

async function loadFaceDb() {
    try {
        const res = await fetch("/api/faces");
        const data = await res.json();
        faceDb = data.faces || [];
        renderFaceList();
    } catch (e) {
        console.error("Failed to load face DB:", e);
    }
}

function renderFaceList() {
    if (faceDb.length === 0) {
        faceListEl.innerHTML = '<li style="color: #a0a0c0">No faces registered</li>';
        return;
    }

    faceListEl.innerHTML = faceDb
        .map(
            (name) => `
        <li>
            <span>${name}</span>
            <button class="btn btn-danger" onclick="deleteFace('${name}')">Delete</button>
        </li>
    `
        )
        .join("");
}

async function deleteFace(name) {
    try {
        await fetch(`/api/faces/${encodeURIComponent(name)}`, { method: "DELETE" });
        await loadFaceDb();
    } catch (e) {
        console.error("Delete failed:", e);
    }
}

function showEnrollDialog() {
    enrollDialog.classList.add("active");
    enrollNameInput.value = "";
    enrollNameInput.focus();
}

function hideEnrollDialog() {
    enrollDialog.classList.remove("active");
}

async function enrollFace() {
    const name = enrollNameInput.value.trim();
    if (!name) return;

    try {
        const res = await fetch(`/api/faces/enroll?name=${encodeURIComponent(name)}`, {
            method: "POST",
        });
        const data = await res.json();
        if (data.status === "enrolled") {
            hideEnrollDialog();
            await loadFaceDb();
        } else {
            alert(data.error || "Enrollment failed");
        }
    } catch (e) {
        alert("Enrollment failed: " + e.message);
    }
}

// Key handler for enroll dialog
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hideEnrollDialog();
    if (e.key === "Enter" && enrollDialog.classList.contains("active")) {
        enrollFace();
    }
});

// --- Init ---

function init() {
    // Set MJPEG stream source
    videoEl.src = "/stream";

    connectWS();
    loadFaceDb();
    requestAnimationFrame(drawOverlay);

    // Refresh face DB periodically
    setInterval(loadFaceDb, 10000);
}

init();
