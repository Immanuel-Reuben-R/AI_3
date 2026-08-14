/* ==========================================================================
   DERMATOSCAN PRO AI - CLIENT APPLICATION SCRIPT
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    
    // ---------------------------------------------------------
    // Application State
    // ---------------------------------------------------------
    const state = {
        currentImageB64: null,
        currentUuid: null,
        processedImageUrl: null,
        heatmapImageUrl: null,
        lastPrediction: null,
        activeView: 'original', // 'original' | 'processed' | 'heatmap'
        webcamStream: null,
    };

    // ---------------------------------------------------------
    // DOM Element References
    // ---------------------------------------------------------
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIconSun = document.getElementById('themeIconSun');
    const themeIconMoon = document.getElementById('themeIconMoon');
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabPanes = document.querySelectorAll('.tab-pane');

    const dropzone = document.getElementById('dropzone');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImgBtn = document.getElementById('removeImgBtn');

    const openCamBtn = document.getElementById('openCamBtn');
    const cameraBox = document.getElementById('cameraBox');
    const webcamVideo = document.getElementById('webcamVideo');
    const snapBtn = document.getElementById('snapBtn');
    const closeCamBtn = document.getElementById('closeCamBtn');

    const sampleChips = document.querySelectorAll('.sample-chip');
    const hairRemovalToggle = document.getElementById('hairRemovalToggle');
    const ttaToggle = document.getElementById('ttaToggle');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValueDisplay = document.getElementById('thresholdValueDisplay');
    const runScanBtn = document.getElementById('runScanBtn');

    const gaugeCircle = document.getElementById('gaugeCircle');
    const probPercentVal = document.getElementById('probPercentVal');
    const riskCategoryPill = document.getElementById('riskCategoryPill');
    const confidenceValue = document.getElementById('confidenceValue');
    const predictionTitle = document.getElementById('predictionTitle');
    const recommendationText = document.getElementById('recommendationText');

    const viewerImage = document.getElementById('viewerImage');
    const viewBtns = document.querySelectorAll('.view-btn');
    const heatmapLegend = document.getElementById('heatmapLegend');

    const abcdeAsymmetryBar = document.getElementById('abcdeAsymmetryBar');
    const abcdeAsymmetryScore = document.getElementById('abcdeAsymmetryScore');
    const abcdeAsymmetryStatus = document.getElementById('abcdeAsymmetryStatus');

    const abcdeBorderBar = document.getElementById('abcdeBorderBar');
    const abcdeBorderScore = document.getElementById('abcdeBorderScore');
    const abcdeBorderStatus = document.getElementById('abcdeBorderStatus');

    const abcdeColorBar = document.getElementById('abcdeColorBar');
    const abcdeColorScore = document.getElementById('abcdeColorScore');
    const abcdeColorStatus = document.getElementById('abcdeColorStatus');

    const abcdeDiameterVal = document.getElementById('abcdeDiameterVal');
    const abcdeDiameterStatus = document.getElementById('abcdeDiameterStatus');

    const abcdeEvolutionStatus = document.getElementById('abcdeEvolutionStatus');

    const latencyPill = document.getElementById('latencyPill');
    const devicePill = document.getElementById('devicePill');
    const hairRemovalPill = document.getElementById('hairRemovalPill');

    const printReportBtn = document.getElementById('printReportBtn');
    const pushEsp32Btn = document.getElementById('pushEsp32Btn');

    const qrCodeImg = document.getElementById('qrCodeImg');
    const localUrlCode = document.getElementById('localUrlCode');
    const espIpInput = document.getElementById('espIpInput');
    const saveEspBtn = document.getElementById('saveEspBtn');

    const oledPred = document.getElementById('oledPred');
    const oledRisk = document.getElementById('oledRisk');
    const oledPct = document.getElementById('oledPct');

    const fetchGeminiBtn = document.getElementById('fetchGeminiBtn');
    const geminiOutput = document.getElementById('geminiOutput');

    // ---------------------------------------------------------
    // Theme Switcher
    // ---------------------------------------------------------
    const savedTheme = localStorage.getItem('dermatoscan_theme') || 'light';
    setTheme(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.body.classList.contains('theme-dark');
        setTheme(isDark ? 'light' : 'dark');
    });

    function setTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.remove('theme-light');
            document.body.classList.add('theme-dark');
            themeIconSun.classList.add('hidden');
            themeIconMoon.classList.remove('hidden');
        } else {
            document.body.classList.remove('theme-dark');
            document.body.classList.add('theme-light');
            themeIconSun.classList.remove('hidden');
            themeIconMoon.classList.add('hidden');
        }
        localStorage.setItem('dermatoscan_theme', theme);
    }

    // ---------------------------------------------------------
    // Navigation Tabs
    // ---------------------------------------------------------
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-tab');
            
            navTabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(targetId).classList.add('active');
        });
    });

    function switchTab(tabId) {
        const tabBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
        if (tabBtn) tabBtn.click();
    }

    // ---------------------------------------------------------
    // Threshold Slider
    // ---------------------------------------------------------
    thresholdSlider.addEventListener('input', (e) => {
        thresholdValueDisplay.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // ---------------------------------------------------------
    // Drag and Drop & Image Upload Handling
    // ---------------------------------------------------------
    browseBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedImage();
    });

    function handleFileSelect(file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
            setImagePreview(evt.target.result);
        };
        reader.readAsDataURL(file);
    }

    function setImagePreview(b64Url) {
        state.currentImageB64 = b64Url;
        imagePreview.src = b64Url;
        viewerImage.src = b64Url;
        
        dropzoneContent.classList.add('hidden');
        cameraBox.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        stopWebcam();
    }

    function clearSelectedImage() {
        state.currentImageB64 = null;
        state.currentUuid = null;
        state.processedImageUrl = null;
        state.heatmapImageUrl = null;
        imagePreview.src = '';
        fileInput.value = '';
        
        previewContainer.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');
    }

    // ---------------------------------------------------------
    // Live Dermatoscope Webcam Stream
    // ---------------------------------------------------------
    openCamBtn.addEventListener('click', async () => {
        try {
            state.webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }
            });
            webcamVideo.srcObject = state.webcamStream;
            dropzoneContent.classList.add('hidden');
            previewContainer.classList.add('hidden');
            cameraBox.classList.remove('hidden');
        } catch (err) {
            alert("Could not access webcam camera: " + err.message);
        }
    });

    snapBtn.addEventListener('click', () => {
        if (!state.webcamStream) return;
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth || 640;
        canvas.height = webcamVideo.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        const b64 = canvas.toDataURL('image/jpeg', 0.92);
        setImagePreview(b64);
    });

    closeCamBtn.addEventListener('click', () => {
        stopWebcam();
        cameraBox.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');
    });

    function stopWebcam() {
        if (state.webcamStream) {
            state.webcamStream.getTracks().forEach(track => track.stop());
            state.webcamStream = null;
        }
    }

    // ---------------------------------------------------------
    // Sample Synthetic Dermoscopy Generators
    // ---------------------------------------------------------
    sampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const type = chip.getAttribute('data-sample');
            generateSampleDermoscopyImage(type);
        });
    });

    function generateSampleDermoscopyImage(type) {
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');

        // Background skin tone
        const bgGrad = ctx.createRadialGradient(200, 200, 50, 200, 200, 200);
        bgGrad.addColorStop(0, '#E8C5A8');
        bgGrad.addColorStop(1, '#D4AB8C');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, 400, 400);

        // Dermoscopy vignette ring
        const ringGrad = ctx.createRadialGradient(200, 200, 160, 200, 200, 200);
        ringGrad.addColorStop(0, 'rgba(0,0,0,0)');
        ringGrad.addColorStop(1, 'rgba(0,0,0,0.85)');
        ctx.fillStyle = ringGrad;
        ctx.fillRect(0, 0, 400, 400);

        // Draw lesion pattern based on type
        if (type === 'melanoma') {
            // Asymmetric malignant melanoma lesion
            ctx.fillStyle = '#2A1810';
            ctx.beginPath();
            ctx.ellipse(200, 200, 85, 55, Math.PI / 4, 0, 2 * Math.PI);
            ctx.fill();

            // Dark malignant hyperpigmented focus
            ctx.fillStyle = '#0D0503';
            ctx.beginPath();
            ctx.ellipse(220, 180, 45, 30, Math.PI / 6, 0, 2 * Math.PI);
            ctx.fill();

            // Scalloped irregular borders
            ctx.fillStyle = '#6E2C1B';
            ctx.beginPath();
            ctx.arc(150, 210, 35, 0, 2 * Math.PI);
            ctx.fill();
        } else if (type === 'nevus') {
            // Uniform benign melanocytic nevus
            ctx.fillStyle = '#5A3D28';
            ctx.beginPath();
            ctx.arc(200, 200, 60, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#422B1B';
            ctx.beginPath();
            ctx.arc(200, 200, 40, 0, 2 * Math.PI);
            ctx.fill();
        } else if (type === 'bcc') {
            // Translucent pearly BCC lesion with telangiectasia
            ctx.fillStyle = '#C47E6A';
            ctx.beginPath();
            ctx.arc(200, 200, 70, 0, 2 * Math.PI);
            ctx.fill();

            // Central erosion
            ctx.fillStyle = '#7A2E20';
            ctx.beginPath();
            ctx.arc(200, 200, 25, 0, 2 * Math.PI);
            ctx.fill();
        } else {
            // Seborrheic keratosis stuck-on warty pattern
            ctx.fillStyle = '#4A3B32';
            ctx.beginPath();
            ctx.arc(200, 200, 75, 0, 2 * Math.PI);
            ctx.fill();

            // Keratin plugs
            ctx.fillStyle = '#1F1713';
            for (let i = 0; i < 8; i++) {
                ctx.beginPath();
                ctx.arc(170 + i*8, 180 + (i%3)*12, 6, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        // Draw synthetic hair occlusions for DullRazor testing
        ctx.strokeStyle = 'rgba(20, 15, 10, 0.7)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(120, 100);
        ctx.quadraticCurveTo(200, 220, 300, 280);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(80, 240);
        ctx.quadraticCurveTo(220, 180, 320, 120);
        ctx.stroke();

        const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
        setImagePreview(dataUrl);
    }

    // ---------------------------------------------------------
    // AI Inference API Request
    // ---------------------------------------------------------
    runScanBtn.addEventListener('click', async () => {
        if (!state.currentImageB64) {
            alert('Please select or upload a skin lesion image first.');
            return;
        }

        runScanBtn.disabled = true;
        runScanBtn.querySelector('span').textContent = 'ANALYZING LESION...';

        try {
            // 1. Upload Pivot
            const resBlob = await fetch(state.currentImageB64);
            const blob = await resBlob.blob();
            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');

            const uploadResp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const uploadData = await uploadResp.json();
            if (!uploadData.success) throw new Error(uploadData.error || "Upload failed");
            
            state.currentUuid = uploadData.uuid;

            // 2. Parallel Fetches
            const use_hair_removal = hairRemovalToggle.checked;
            
            const [infResp, heatResp] = await Promise.all([
                fetch('/api/inference', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        uuid: state.currentUuid,
                        use_tta: ttaToggle.checked,
                        use_hair_removal: use_hair_removal,
                        threshold: parseFloat(thresholdSlider.value)
                    })
                }),
                fetch('/api/heatmap', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        uuid: state.currentUuid,
                        use_processed: use_hair_removal
                    })
                })
            ]);
            
            const infData = await infResp.json();
            const heatData = await heatResp.json();
            
            if (!infData.success) throw new Error(infData.error || "Inference failed");

            // 3. Sequential Fetch for ABCDE
            const abcdeResp = await fetch('/api/abcde', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    uuid: state.currentUuid,
                    probability: infData.probability
                })
            });
            const abcdeData = await abcdeResp.json();

            const data = {
                ...infData,
                heatmap_url: heatData.heatmap_url,
                abcde: abcdeData.abcde
            };

            state.lastPrediction = data;
            state.processedImageUrl = infData.processed_url ? infData.processed_url : null;
            state.heatmapImageUrl = heatData.heatmap_url ? heatData.heatmap_url : null;

            updateDashboardUI(data);
            switchTab('tab-dashboard');

        } catch (err) {
            alert('Error running AI inference: ' + err.message);
        } finally {
            runScanBtn.disabled = false;
            runScanBtn.querySelector('span').textContent = 'RUN AI SKIN CANCER SCAN';
        }
    });

    // ---------------------------------------------------------
    // Update Dashboard UI
    // ---------------------------------------------------------
    function updateDashboardUI(data) {
        const probPct = data.probability_percent || 0;
        
        // Gauge circle animation (Circumference = 314.15)
        const offset = 314.15 - (314.15 * probPct / 100);
        gaugeCircle.style.strokeDashoffset = offset;
        probPercentVal.textContent = `${probPct.toFixed(1)}%`;

        // Prediction heading & pill
        predictionTitle.textContent = data.prediction || 'Likely Benign';
        recommendationText.textContent = data.recommendation || '';
        confidenceValue.textContent = `Confidence: ${data.confidence || 90}%`;

        // Risk pill style
        riskCategoryPill.className = `pill-badge ${(data.risk_code || 'low').toLowerCase()}`;
        riskCategoryPill.textContent = data.risk_level || 'LOW RISK';

        // Meta tags
        latencyPill.textContent = `Latency: ${data.latency_ms || 0} ms`;
        devicePill.textContent = `Device: ${data.device || 'PyTorch'}`;
        hairRemovalPill.textContent = `DullRazor: ${data.hair_removal_applied ? 'Applied' : 'Disabled'}`;

        // Initial view image
        state.activeView = 'original';
        updateViewerImage();

        // Update ABCDE scores
        if (data.abcde) {
            const a = data.abcde.asymmetry;
            abcdeAsymmetryScore.textContent = `${a.score}/10`;
            abcdeAsymmetryStatus.textContent = a.level;
            abcdeAsymmetryBar.style.width = `${a.score * 10}%`;

            const b = data.abcde.border;
            abcdeBorderScore.textContent = `${b.score}/10`;
            abcdeBorderStatus.textContent = b.level;
            abcdeBorderBar.style.width = `${b.score * 10}%`;

            const c = data.abcde.color;
            abcdeColorScore.textContent = `${c.score}/10`;
            abcdeColorStatus.textContent = c.level;
            abcdeColorBar.style.width = `${c.score * 10}%`;

            const d = data.abcde.diameter;
            abcdeDiameterVal.textContent = d.value;
            abcdeDiameterStatus.textContent = d.level;

            const e = data.abcde.evolution;
            abcdeEvolutionStatus.textContent = e.status;
        }

        // Push to OLED Simulator
        oledPred.textContent = data.prediction.toUpperCase();
        oledRisk.textContent = `[ ${data.risk_code} ]`;
        oledPct.textContent = `${probPct.toFixed(1)}%`;
    }

    // ---------------------------------------------------------
    // View Switcher (Original / Hair Removed / Grad-CAM)
    // ---------------------------------------------------------
    viewBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            viewBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeView = btn.getAttribute('data-view');
            updateViewerImage();
        });
    });

    function updateViewerImage() {
        if (state.activeView === 'processed' && state.processedImageUrl) {
            viewerImage.src = state.processedImageUrl;
            heatmapLegend.classList.add('hidden');
        } else if (state.activeView === 'heatmap' && state.heatmapImageUrl) {
            viewerImage.src = state.heatmapImageUrl;
            heatmapLegend.classList.remove('hidden');
        } else {
            viewerImage.src = state.currentImageB64 || viewerImage.src;
            heatmapLegend.classList.add('hidden');
        }
    }

    // ---------------------------------------------------------
    // Network Info & QR Code Initialization
    // ---------------------------------------------------------
    async function loadNetworkInfo() {
        try {
            const resp = await fetch('/api/network-info');
            const data = await resp.json();
            if (data.success) {
                if (data.qr_code_b64) qrCodeImg.src = data.qr_code_b64;
                localUrlCode.textContent = data.local_url;
            }
        } catch (e) {
            console.log("Network info load note:", e);
        }
    }
    loadNetworkInfo();

    // ---------------------------------------------------------
    // ESP32 Connection
    // ---------------------------------------------------------
    saveEspBtn.addEventListener('click', async () => {
        const ip = espIpInput.value.trim();
        if (!ip) return;
        try {
            const resp = await fetch('/api/esp32/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ esp32_ip: ip })
            });
            const data = await resp.json();
            if (data.success) alert(`ESP32 IP configured: ${ip}`);
        } catch (e) {
            alert('Failed connecting ESP32: ' + e.message);
        }
    });

    pushEsp32Btn.addEventListener('click', async () => {
        if (!state.lastPrediction) {
            alert('Run a scan first to push results.');
            return;
        }
        alert(`Pushed ${state.lastPrediction.prediction} (${state.lastPrediction.probability_percent}%) to ESP32 OLED Screen.`);
    });

    // ---------------------------------------------------------
    // Gemini AI Consultation
    // ---------------------------------------------------------
    fetchGeminiBtn.addEventListener('click', async () => {
        if (!state.currentUuid) {
            alert('Please select and scan an image first.');
            return;
        }

        geminiOutput.innerHTML = '<div class="gemini-placeholder">Analyzing lesion morphological patterns with Gemini Vision AI...</div>';

        try {
            const resp = await fetch('/api/analyze-gemini', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uuid: state.currentUuid })
            });
            const data = await resp.json();
            if (data.success) {
                geminiOutput.innerHTML = formatMarkdown(data.analysis);
            }
        } catch (err) {
            geminiOutput.innerHTML = `<p style="color: var(--status-danger);">Error fetching Gemini Vision opinion: ${err.message}</p>`;
        }
    });

    function formatMarkdown(text) {
        return text
            .replace(/^### (.*$)/gim, '<h3 style="font-size: 1.1rem; color: var(--maroon-primary); margin-top: 14px; margin-bottom: 6px;">$1</h3>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/^- (.*$)/gim, '<li style="margin-left: 18px;">$1</li>')
            .replace(/\n/g, '<br>');
    }

    // ---------------------------------------------------------
    // PDF Report Print
    // ---------------------------------------------------------
    printReportBtn.addEventListener('click', () => {
        window.print();
    });

});
