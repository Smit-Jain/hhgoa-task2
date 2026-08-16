const micBtn = document.getElementById('mic-btn');
const statusText = document.getElementById('status-text');
const outputSection = document.getElementById('output-section');
const transcriptionText = document.getElementById('transcription-text');
const answerText = document.getElementById('answer-text');
const latencyMetric = document.getElementById('latency-metric');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

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
        console.error("Error accessing microphone:", err);
        statusText.textContent = "Microphone access denied.";
        micBtn.style.opacity = '0.5';
        micBtn.style.cursor = 'not-allowed';
    }
}

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
        statusText.textContent = "Processing...";
    } else {
        mediaRecorder.start();
        isRecording = true;
        micBtn.classList.add('recording');
        statusText.textContent = "Listening (click to stop)...";
        outputSection.classList.add('hidden');
    }
});

async function sendAudioToBackend(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            transcriptionText.textContent = data.query;
            answerText.textContent = data.answer;
            
            // Format metrics nicely
            if (data.total_latency_ms) {
                latencyMetric.textContent = `Latency: ${data.total_latency_ms} ms | Safe: ${data.metadata?.is_safe} | Hallucination Detected: ${data.metadata?.hallucination_detected}`;
            } else {
                latencyMetric.textContent = '';
            }
            
            outputSection.classList.remove('hidden');
        } else {
            alert(data.error || "An error occurred during processing.");
            statusText.textContent = "Error occurred.";
        }
    } catch (err) {
        console.error("Fetch error:", err);
        alert("Failed to connect to the server.");
        statusText.textContent = "Connection failed.";
    } finally {
        micBtn.classList.remove('processing');
        if (!isRecording) {
            statusText.textContent = "Click to speak";
        }
    }
}

// Initialize on page load
window.addEventListener('load', setupAudio);
