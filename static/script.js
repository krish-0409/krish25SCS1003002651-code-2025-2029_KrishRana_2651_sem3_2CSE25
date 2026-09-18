const dropZone = document.getElementById("dropZone");
const dropZoneContent = document.getElementById("dropZoneContent");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("errorBox");
const results = document.getElementById("results");
const topLabel = document.getElementById("topLabel");
const topConfidenceBar = document.getElementById("topConfidenceBar");
const topConfidenceText = document.getElementById("topConfidenceText");
const otherResults = document.getElementById("otherResults");

let selectedFile = null;

// ---------- Helpers ----------
function resetUI() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.hidden = true;
  previewImg.src = "";
  dropZoneContent.hidden = false;
  predictBtn.disabled = true;
  resetBtn.hidden = true;
  hideError();
  results.hidden = true;
  otherResults.innerHTML = "";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function handleFileSelect(file) {
  hideError();
  results.hidden = true;

  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showError("Please select a valid image file.");
    return;
  }

  const maxBytes = 8 * 1024 * 1024;
  if (file.size > maxBytes) {
    showError("File is too large. Maximum allowed size is 8 MB.");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropZoneContent.hidden = true;
  };
  reader.readAsDataURL(file);

  predictBtn.disabled = false;
  resetBtn.hidden = false;
}

// ---------- Event listeners ----------
dropZone.addEventListener("click", () => {
  if (!selectedFile) fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  handleFileSelect(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  handleFileSelect(file);
});

resetBtn.addEventListener("click", resetUI);

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  results.hidden = true;
  loading.hidden = false;
  predictBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (parseErr) {
      throw new Error("Server returned an unexpected response.");
    }

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Something went wrong while classifying the image.");
    }

    renderResults(data);
  } catch (err) {
    showError(err.message || "Network error — please try again.");
  } finally {
    loading.hidden = true;
    predictBtn.disabled = false;
  }
});

function renderResults(data) {
  const top = data.top_prediction;

  topLabel.textContent = top.label;
  topConfidenceText.textContent = `${top.confidence}% confidence`;
  topConfidenceBar.style.width = "0%";
  requestAnimationFrame(() => {
    topConfidenceBar.style.width = `${top.confidence}%`;
  });

  otherResults.innerHTML = "";
  data.predictions.slice(1).forEach((pred) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${pred.label}</span><span class="conf">${pred.confidence}%</span>`;
    otherResults.appendChild(li);
  });

  results.hidden = false;
}
