/**
 * main.js — Entry point
 * Wires together upload, webcam, pose detection, color extraction,
 * scoring, and result rendering.
 */
import './style.css';
import { initDetector, detectPose, deriveClothingZones } from './src/poseDetector.js';
import { extractZoneColors } from './src/colorExtractor.js';
//import { scoreOutfit } from './src/outfitScorer.js';
import { scoreOutfit, initMLScorer } from './src/outfitScorerML.js';
import { startCamera, stopCamera, captureFrame, isCameraActive } from './src/webcam.js';
import {
    animateScore, renderItems, renderBreakdown,
    renderTips, setVerdict, setLoadingStep, completeAllSteps,
} from './src/ui.js';

// ── DOM Elements ──
const tabUpload = document.getElementById('tab-upload');
const tabWebcam = document.getElementById('tab-webcam');
const panelUpload = document.getElementById('panel-upload');
const panelWebcam = document.getElementById('panel-webcam');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

const webcamVideo = document.getElementById('webcam-video');
const webcamPlaceholder = document.getElementById('webcam-placeholder');
const btnStartCam = document.getElementById('btn-start-cam');
const btnCapture = document.getElementById('btn-capture');
const btnStopCam = document.getElementById('btn-stop-cam');

const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const previewCanvas = document.getElementById('preview-canvas');
const analysisCanvas = document.getElementById('analysis-canvas');
const btnAnalyze = document.getElementById('btn-analyze');
const btnReset = document.getElementById('btn-reset');

const inputSection = document.getElementById('input-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const btnTryAgain = document.getElementById('btn-try-again');

// ── Tab Switching ──
tabUpload.addEventListener('click', () => switchTab('upload'));
tabWebcam.addEventListener('click', () => switchTab('webcam'));

function switchTab(tab) {
    tabUpload.classList.toggle('active', tab === 'upload');
    tabWebcam.classList.toggle('active', tab === 'webcam');
    panelUpload.classList.toggle('active', tab === 'upload');
    panelWebcam.classList.toggle('active', tab === 'webcam');

    if (tab === 'upload' && isCameraActive()) {
        stopCamera(webcamVideo);
        webcamPlaceholder.classList.remove('hidden');
        btnStartCam.disabled = false;
        btnCapture.disabled = true;
        btnStopCam.disabled = true;
    }
}

// ── File Upload ──
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageFile(file);
});

function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => showPreview(e.target.result);
    reader.readAsDataURL(file);
}

// ── Webcam ──
btnStartCam.addEventListener('click', async () => {
    const ok = await startCamera(webcamVideo);
    if (ok) {
        webcamPlaceholder.classList.add('hidden');
        btnStartCam.disabled = true;
        btnCapture.disabled = false;
        btnStopCam.disabled = false;
    } else {
        alert('Could not access camera. Please allow camera permissions and try again.');
    }
});

btnCapture.addEventListener('click', () => {
    const dataUrl = captureFrame(webcamVideo, previewCanvas);
    if (dataUrl) {
        stopCamera(webcamVideo);
        webcamPlaceholder.classList.remove('hidden');
        btnStartCam.disabled = false;
        btnCapture.disabled = true;
        btnStopCam.disabled = true;
        showPreview(dataUrl);
    }
});

btnStopCam.addEventListener('click', () => {
    stopCamera(webcamVideo);
    webcamPlaceholder.classList.remove('hidden');
    btnStartCam.disabled = false;
    btnCapture.disabled = true;
    btnStopCam.disabled = true;
});

// ── Preview ──
function showPreview(dataUrl) {
    previewImage.src = dataUrl;
    previewSection.classList.remove('hidden');
    inputSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
}

btnReset.addEventListener('click', resetToInput);
btnTryAgain.addEventListener('click', resetToInput);

function resetToInput() {
    previewSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingSection.classList.add('hidden');
    inputSection.classList.remove('hidden');
    fileInput.value = '';

    // Reset loading steps
    document.querySelectorAll('.loader-steps .step').forEach(el => {
        el.classList.remove('active', 'done');
    });
}

// ── Analysis Pipeline ──
btnAnalyze.addEventListener('click', runAnalysis);

async function runAnalysis() {
    previewSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    try {
        // 1. Load image onto analysis canvas
        setLoadingStep('step-pose');
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = previewImage.src;
        });

        analysisCanvas.width = img.naturalWidth;
        analysisCanvas.height = img.naturalHeight;
        const ctx = analysisCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        // 2. Detect pose
        await initDetector();
        const pose = await detectPose(analysisCanvas);

        if (!pose) {
            throw new Error('NO_PERSON');
        }

        // 3. Derive zones
        setLoadingStep('step-zones');
        await sleep(300); // brief pause for UX
        const zones = deriveClothingZones(pose, img.naturalWidth, img.naturalHeight);

        if (Object.keys(zones).length === 0) {
            throw new Error('NO_ZONES');
        }

        // 4. Extract colors
        setLoadingStep('step-colors');
        await sleep(300);

        const zoneColors = {};
        for (const [zoneName, zoneBox] of Object.entries(zones)) {
            zoneColors[zoneName] = extractZoneColors(ctx, zoneBox);
        }

        // 5. Score outfit
        setLoadingStep('step-score');
        await sleep(400);

        const result = await scoreOutfit(zoneColors);

        completeAllSteps();
        await sleep(500);

        // 6. Show results
        showResults(result, zoneColors);
    } catch (err) {
        console.error('Analysis error:', err);
        loadingSection.classList.add('hidden');

        if (err.message === 'NO_PERSON') {
            alert('😕 Could not detect a person in this image. Please try a clearer full-body photo.');
        } else if (err.message === 'NO_ZONES') {
            alert('😕 Could not identify clothing zones. Please try a front-facing full-body photo.');
        } else {
            alert('Something went wrong during analysis. Please try another photo.');
        }

        resetToInput();
    }
}

function showResults(result, zoneColors) {
    loadingSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    resultsSection.style.animation = 'none';
    resultsSection.offsetHeight; // force reflow
    resultsSection.style.animation = 'fadeIn 0.6s ease';

    console.log('=== SHOWING RESULTS ===');
    console.log('Result object:', result);
    console.log('Total score:', result.totalScore);
    
    // Make sure score is a number
    const scoreToDisplay = result.totalScore || 0;
    console.log('Score to animate:', scoreToDisplay);
    
    // Animate score
    animateScore(scoreToDisplay);
    
    // Set verdict
    if (result.verdict) {
        setVerdict(result.verdict);
    }
    
    // Render items
    renderItems(zoneColors);
    
    // Render breakdown
    if (result.breakdown) {
        renderBreakdown(result.breakdown);
    }
    
    // Render tips
    if (result.tips && Array.isArray(result.tips)) {
        renderTips(result.tips);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Pre-load models in background ──
initDetector().catch(err => {
    console.warn('Pose model pre-load failed, will retry on first analysis:', err);
});

initMLScorer().catch(err => {
    console.warn('ML model pre-load failed, will retry on first analysis:', err);
});
