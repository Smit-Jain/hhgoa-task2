const micBtn = document.getElementById('mic-btn');
const statusText = document.getElementById('status-text');
const waveContainer = document.getElementById('wave-container');
const transcriptionText = document.getElementById('transcription-text');
const answerText = document.getElementById('answer-text');
const latencyVal = document.getElementById('latency-val');
const badgeSafety = document.getElementById('badge-safety');
const badgeSafetyText = document.getElementById('badge-safety-text');
const badgeGrounding = document.getElementById('badge-grounding');
const badgeGroundingText = document.getElementById('badge-grounding-text');
const toggleContexts = document.getElementById('toggle-contexts');
const contextsBody = document.getElementById('contexts-body');
const contextsList = document.getElementById('contexts-list');
const contextCount = document.getElementById('context-count');
const chevron = document.getElementById('chevron');
const signArrows = document.querySelectorAll('.sign-arrow');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Request microphone access
async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];
            await sendAudioToBackend(audioBlob);
        };
    } catch (err) {
        console.error("Microphone permission error:", err);
        statusText.textContent = "ERR: MIC ACCESS DENIED";
        micBtn.style.opacity = '0.4';
        micBtn.style.cursor = 'not-allowed';
    }
}

// Mic button toggle
micBtn.addEventListener('click', () => {
    if (!mediaRecorder) {
        alert("Microphone not available or permission denied.");
        return;
    }
    
    if (isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        micBtn.classList.remove('recording');
        micBtn.classList.add('processing');
        waveContainer.classList.remove('active');
        statusText.textContent = "PROCESSING SPEECH...";
    } else {
        mediaRecorder.start();
        isRecording = true;
        micBtn.classList.add('recording');
        waveContainer.classList.add('active');
        statusText.textContent = "LISTENING... (CLICK TO STOP)";
    }
});

// Toggle Contexts Drawer
toggleContexts.addEventListener('click', () => {
    const isHidden = contextsBody.classList.toggle('hidden');
    chevron.textContent = isHidden ? '+' : '−';
});

// Preset Signpost Query Buttons
signArrows.forEach(btn => {
    btn.addEventListener('click', async () => {
        const queryText = btn.getAttribute('data-query');
        statusText.textContent = `TESTING: "${queryText.substring(0, 20)}..."`;
        micBtn.classList.add('processing');
        
        const audioBlob = createSilentAudioBlob();
        await sendAudioToBackendWithOverride(audioBlob, queryText);
    });
});

function createSilentAudioBlob() {
    const wavHeader = new Uint8Array([
        0x52, 0x49, 0x46, 0x46, 0x24, 0x08, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
        0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x44, 0xac, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00, 0x02, 0x00, 0x10, 0x00,
        0x64, 0x61, 0x74, 0x61, 0x00, 0x08, 0x00, 0x00
    ]);
    const silence = new Uint8Array(2048);
    return new Blob([wavHeader, silence], { type: 'audio/wav' });
}

async function sendAudioToBackendWithOverride(audioBlob, overrideQuery) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    if (overrideQuery) {
        formData.append('override_query', overrideQuery);
    }
    await executeFetch(formData);
}

async function sendAudioToBackend(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    await executeFetch(formData);
}

async function executeFetch(formData) {
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            renderOutput(data);
        } else {
            alert(data.error || "An error occurred during pipeline execution.");
            statusText.textContent = "ERR: EXECUTION FAILED";
        }
    } catch (err) {
        console.error("Fetch error:", err);
        statusText.textContent = "ERR: CONNECTION FAILED";
    } finally {
        micBtn.classList.remove('processing');
        if (!isRecording) {
            statusText.textContent = "CLICK TO SPEAK";
        }
    }
}

function renderOutput(data) {
    transcriptionText.textContent = data.query || "[NO_QUERY_CAPTURED]";
    answerText.textContent = data.answer || "[NO_ANSWER_GENERATED]";
    
    // Latency
    const latency = data.total_end_to_end_latency_ms || data.total_latency_ms || 0;
    latencyVal.textContent = `${latency} ms`;
    
    // Telemetry Badges
    const isSafe = data.metadata?.is_safe !== false;
    const isOffTopic = data.metadata?.is_off_topic === true;
    const isHallucination = data.metadata?.hallucination_detected === true;
    
    // Safety Badge
    if (isSafe) {
        badgeSafety.className = 'tele-badge badge-green';
        badgeSafetyText.textContent = 'PASSED';
    } else {
        badgeSafety.className = 'tele-badge badge-red';
        badgeSafetyText.textContent = 'VIOLATION';
    }
    
    // Grounding Badge
    if (isOffTopic) {
        badgeGrounding.className = 'tele-badge badge-amber';
        badgeGroundingText.textContent = 'OFF_TOPIC';
    } else if (isHallucination) {
        badgeGrounding.className = 'tele-badge badge-amber';
        badgeGroundingText.textContent = 'UNGROUNDED';
    } else {
        badgeGrounding.className = 'tele-badge badge-green';
        badgeGroundingText.textContent = 'GROUNDED';
    }
    
    // Passages
    contextsList.innerHTML = '';
    const contexts = data.retrieved_contexts || [];
    contextCount.textContent = contexts.length;
    
    if (contexts.length === 0) {
        contextsList.innerHTML = '<div class="empty-msg">No passages matched in Qdrant index.</div>';
    } else {
        contexts.forEach((ctx, idx) => {
            const item = document.createElement('div');
            item.className = 'passage-card';
            item.innerHTML = `<strong>[PASSAGE_${idx + 1}]</strong> ${ctx}`;
            contextsList.appendChild(item);
        });
    }
}

// Initialize on page load
window.addEventListener('load', setupAudio);
