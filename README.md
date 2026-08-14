# SKIN CANCER WEB UI - DermatoScan Pro AI

This directory contains the complete web interface and PyTorch backend engine for Skin Cancer AI detection.

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the AI backend server:
   ```bash
   python app.py
   ```

3. Open your browser at:
   - Local: `http://127.0.0.1:8080`
   - Mobile / LAN: `http://<your-ip>:8080`

## Features

- **Trained Model**: Connected directly to `F:\BEST RUN\artifacts\skin_cancer_model.pth`
- **Medical Theme**: Neutral clinical aesthetic with Maroon & Burgundy palette
- **Preprocessing**: DullRazor hair artifact filter & Test-Time Augmentation (TTA)
- **Grad-CAM Visual Heatmaps**: Explainable AI attention maps overlaid on skin lesions
- **ABCDE Criteria**: Automatic dermatological feature assessment
- **ESP32 & Mobile Sync**: Real-time OLED display streaming & QR code mobile pairing
- **Gemini AI Vision Integration**: Instant second-opinion clinical explanations
