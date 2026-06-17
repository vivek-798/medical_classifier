// Configuration
let API_BASE_URL = fetch("https://medical-classifier-backend-3u8j.onrender.com/classify"); // Fallback URL

const MEDICAL_CATEGORIES = {
    "cbc": "CBC (Complete Blood Count)",
    "crp": "CRP (C-Reactive Protein)",
    "lft": "LFT (Liver Function Test)",
    "kidney_function_test": "Kidney Function Test / KFT",
    "urine": "Urine Analysis",
    "microbiology": "Microbiology Culture",
    "haematology": "Haematology",
    "lipid_profile": "Lipid Profile",
    "thyroid_profile": "Thyroid Profile",
    "electrolytes": "Electrolytes",
    "widel": "Widal Test",
    "rbs": "RBS (Random Blood Sugar)",
    "blood_grounping_rh": "Blood Grouping & Rh",
    "vitamin_d": "Vitamin D",
    "vitamin_b12": "Vitamin B12",
    "abg": "ABG (Arterial Blood Gas)",
    "coagulation": "Coagulation",
    "unknown": "Unknown / Other"
};

// Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneContent = document.getElementById("dropzoneContent");
const filePreview = document.getElementById("filePreview");
const fileNameSpan = document.getElementById("fileName");
const fileIcon = document.getElementById("fileIcon");
const clearFileBtn = document.getElementById("clearFile");
const uploadForm = document.getElementById("uploadForm");
const submitBtn = document.getElementById("submitBtn");
const btnSpinner = document.getElementById("btnSpinner");
const btnText = document.getElementById("btnText");

// Results Elements
const resultsCard = document.getElementById("resultsCard");
const predictedCategoryValue = document.getElementById("predictedCategoryValue");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceProgressFill = document.getElementById("confidenceProgressFill");
const metaFileName = document.getElementById("metaFileName");
const keywordTagsContainer = document.getElementById("keywordTagsContainer");
const ocrTextContent = document.getElementById("ocrTextContent");
const copyBtn = document.getElementById("copyBtn");

// Error Elements
const errorCard = document.getElementById("errorCard");
const errorMessage = document.getElementById("errorMessage");

// Feedback Elements
const feedbackSection = document.getElementById("feedbackSection");
const feedbackButtons = document.getElementById("feedbackButtons");
const feedbackCorrectBtn = document.getElementById("feedbackCorrectBtn");
const feedbackIncorrectBtn = document.getElementById("feedbackIncorrectBtn");
const feedbackIncorrectForm = document.getElementById("feedbackIncorrectForm");
const correctCategorySelect = document.getElementById("correctCategorySelect");
const submitCorrectionBtn = document.getElementById("submitCorrectionBtn");
const feedbackSuccessMsg = document.getElementById("feedbackSuccessMsg");

let currentReportId = null;
let currentPredictedCategory = null;

// Initialize app config
async function initConfig() {
    try {
        const response = await fetch("config.json");
        if (response.ok) {
            const config = await response.json();
            if (config.API_BASE_URL) {
                API_BASE_URL = config.API_BASE_URL.replace(/\/$/, ""); // Trim trailing slash
                console.log("Configured API Base URL:", API_BASE_URL);
            }
        }
    } catch (e) {
        console.warn("Could not load config.json, using default API base URL:", API_BASE_URL);
    }
}

// Setup correction select options
function setupCategoryDropdown() {
    correctCategorySelect.innerHTML = "";
    Object.entries(MEDICAL_CATEGORIES).forEach(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        correctCategorySelect.appendChild(option);
    });
}

// Drag and drop frontend logic
function handleFileSelect(file) {
    if (!file) return;
    fileNameSpan.textContent = file.name;
    
    // Set icon based on extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'pdf') {
        fileIcon.textContent = '📕';
    } else if (['png', 'jpg', 'jpeg', 'tif', 'tiff'].includes(ext)) {
        fileIcon.textContent = '🖼️';
    } else {
        fileIcon.textContent = '📄';
    }
    
    dropzoneContent.style.display = 'none';
    filePreview.style.display = 'flex';
    dropzone.classList.add('has-file');
}

fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag events
['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
    }, false);
});

dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect(files[0]);
    }
});

// Clear file selection
clearFileBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent trigger file input click
    resetFileForm();
});

function resetFileForm() {
    fileInput.value = '';
    filePreview.style.display = 'none';
    dropzoneContent.style.display = 'flex';
    dropzone.classList.remove('has-file');
}

// Reset UI cards
function clearOutputCards() {
    resultsCard.style.display = 'none';
    errorCard.style.display = 'none';
}

// Submit form file upload
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) return;
    
    // UI state: loading
    submitBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnText.textContent = 'Analyzing Report...';
    clearOutputCards();

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let errorText = `Server returned status ${response.status}`;
            try {
                const errJson = await response.json();
                if (errJson.detail) errorText = errJson.detail;
            } catch (pErr) {}
            throw new Error(errorText);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        console.error("Analysis Error:", err);
        renderError(err.message || "An unexpected error occurred during classification.");
    } finally {
        // UI state: idle
        submitBtn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.textContent = 'Analyze Report';
    }
});

// Render results
function renderResults(data) {
    currentReportId = data.report_id;
    currentPredictedCategory = data.category;

    // Fill elements
    const formattedCategory = (MEDICAL_CATEGORIES[data.category] || data.category.replace(/_/g, ' ')).toUpperCase();
    predictedCategoryValue.textContent = formattedCategory;
    
    const confPct = Math.round(data.confidence * 100);
    confidenceValue.textContent = `${confPct}%`;
    confidenceProgressFill.style.style = `width: ${confPct}%`;
    confidenceProgressFill.style.width = `${confPct}%`;
    metaFileName.textContent = data.filename;

    // Keywords
    keywordTagsContainer.innerHTML = "";
    if (!data.matched_keywords || data.matched_keywords.length === 0) {
        const span = document.createElement("span");
        span.className = "tag";
        span.textContent = "None";
        keywordTagsContainer.appendChild(span);
    } else {
        data.matched_keywords.forEach(kw => {
            const span = document.createElement("span");
            span.className = "tag";
            span.textContent = kw;
            keywordTagsContainer.appendChild(span);
        });
    }

    // OCR Text
    ocrTextContent.textContent = data.text || "No text extracted.";

    // Reset feedback UI
    feedbackButtons.style.display = 'flex';
    feedbackIncorrectForm.style.display = 'none';
    feedbackSuccessMsg.style.display = 'none';
    correctCategorySelect.value = data.category in MEDICAL_CATEGORIES ? data.category : 'unknown';

    // Show Card
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

// Render error
function renderError(msg) {
    errorMessage.textContent = msg;
    errorCard.style.display = 'block';
    errorCard.scrollIntoView({ behavior: 'smooth' });
}

// Copy Clipboard logic
copyBtn.addEventListener('click', () => {
    const text = ocrTextContent.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.classList.remove('copied');
        }, 2000);
    });
});

// Feedback submissions
async function sendFeedback(userConfirmation, finalCorrectLabel) {
    if (!currentReportId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/feedback`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                report_id: currentReportId,
                user_confirmation: userConfirmation,
                final_correct_label: finalCorrectLabel
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to save feedback: status ${response.status}`);
        }

        feedbackButtons.style.display = 'none';
        feedbackIncorrectForm.style.display = 'none';
        feedbackSuccessMsg.style.display = 'block';
    } catch (err) {
        console.error("Feedback error:", err);
        alert(`Failed to submit feedback: ${err.message}`);
    }
}

feedbackCorrectBtn.addEventListener('click', () => {
    sendFeedback('correct', currentPredictedCategory);
});

feedbackIncorrectBtn.addEventListener('click', () => {
    feedbackIncorrectForm.style.display = 'flex';
});

submitCorrectionBtn.addEventListener('click', () => {
    const selectedLabel = correctCategorySelect.value;
    sendFeedback('incorrect', selectedLabel);
});

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    initConfig();
    setupCategoryDropdown();
});
