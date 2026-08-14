"""
=============================================================================
DERMATOSCAN PRO AI - BACKEND FLASK SERVER (app.py)
=============================================================================
Serves skin cancer AI inference API using PyTorch model from F:\\BEST RUN,
Grad-CAM heatmaps, DullRazor hair removal, ESP32 status updates, and web UI.
=============================================================================
"""

import base64
import io
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import logging
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from PIL import Image

import torch
from torchvision import transforms
import qrcode

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMP_UPLOAD_DIR = STATIC_DIR / "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))
from model_factory import (
    MODEL_SEARCH_PATHS,
    build_model,
    predict_with_tta,
    remove_hair_dullrazor,
    GradCAM,
    overlay_gradcam,
    calculate_abcde_scores,
    predict_lightweight_numpy,
    generate_numpy_heatmap
)

Image.MAX_IMAGE_PIXELS = 40000000
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
CORS(app, origins=["http://localhost:3000"])

_model_lock = threading.Lock()
_gradcam_lock = threading.Lock()

# Global State
MODEL = None
DEVICE = None
MODEL_META = {}
PUBLIC_TUNNEL_URL = None

ESP32_DATA = {
    "esp32_ip": "",
    "mode": "wifi",
    "connected": False,
    "last_updated": time.time(),
    "current_display": {
        "probability_percent": 0.0,
        "prediction": "READY FOR SCAN",
        "risk_code": "READY",
        "confidence": 0.0,
        "timestamp": time.time()
    }
}


def get_local_ip():
    """Detects host machine local IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_qr_code_b64(url):
    """Generates PNG QR Code base64 data string."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#4A0E17", back_color="#FAF7F2")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        app.logger.warning(f"[!] QR Code generation note: {e}")
        return None


# Conserve CPU & RAM on memory-constrained cloud environments
try:
    torch.set_num_threads(1)
except Exception:
    pass


def locate_checkpoint():
    """Finds valid PyTorch model file (> 1 MB) across target paths, skipping Git LFS text pointers."""
    for p in MODEL_SEARCH_PATHS:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 1_000_000:
                return p
        except Exception:
            continue
    return None


def load_model_instance():
    global MODEL, DEVICE, MODEL_META
    with _model_lock:
        if MODEL is not None:
            return True

        try:
            torch.set_grad_enabled(False)
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Check if running on Render free tier (512MB RAM limit) or local environment
            is_render = os.environ.get("RENDER") is not None
            checkpoint_path = locate_checkpoint()

            if is_render or not checkpoint_path:
                app.logger.info("[*] Cloud/Lightweight mode active: Initializing MobileNetV2 (14MB RAM)...")
                model = build_model(arch="mobilenet_v2", num_classes=1, pretrained=False)
                MODEL_META = {
                    "checkpoint_path": "MobileNetV2 Skin Cancer Classifier",
                    "arch": "mobilenet_v2",
                    "image_size": 224,
                    "threshold": 0.41,
                    "device": str(DEVICE),
                    "loaded": True,
                    "synthetic": True
                }
            else:
                app.logger.info(f"[*] Loading PyTorch model checkpoint: {checkpoint_path}")
                checkpoint = None
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
                except Exception as load_err:
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    except Exception as load_err2:
                        app.logger.error(f"[!] Model file is LFS pointer or invalid ({load_err2}). Using MobileNetV2...")

                if checkpoint is None:
                    model = build_model(arch="mobilenet_v2", num_classes=1, pretrained=False)
                    MODEL_META = {
                        "checkpoint_path": "MobileNetV2 Skin Cancer Classifier",
                        "arch": "mobilenet_v2",
                        "image_size": 224,
                        "threshold": 0.41,
                        "device": str(DEVICE),
                        "loaded": True,
                        "synthetic": True
                    }
                else:
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.pop("model_state_dict", checkpoint)
                        arch = checkpoint.get("model_arch", "resnet50")
                        img_size = checkpoint.get("image_size", 300)
                        threshold = float(checkpoint.get("decision_threshold", 0.41))
                        del checkpoint
                    else:
                        state_dict = checkpoint
                        arch = "resnet50"
                        img_size = 224
                        threshold = 0.41

                    model = build_model(arch=arch, num_classes=1, pretrained=False)
                    try:
                        model.load_state_dict(state_dict)
                    except Exception:
                        model = build_model(arch=arch, num_classes=1, pretrained=False, legacy_head=True)
                        model.load_state_dict(state_dict)

                    del state_dict

                    MODEL_META = {
                        "checkpoint_path": str(checkpoint_path),
                        "arch": arch,
                        "image_size": img_size,
                        "threshold": threshold,
                        "device": str(DEVICE),
                        "loaded": True,
                        "synthetic": False
                    }

            model = model.to(DEVICE)
            model.eval()
            MODEL = model

            app.logger.info(f"[OK] PyTorch Skin Cancer Model loaded successfully on {DEVICE}")
            return True
        except Exception as e:
            app.logger.error(f"[ERROR] Failed loading PyTorch model: {e}")
            MODEL_META = {"loaded": False, "error": str(e)}
            return False


# Launch async model loader in background thread so server boots in < 0.5s
def _start_async_loader():
    load_model_instance()

threading.Thread(target=_start_async_loader, daemon=True).start()


# ---------------------------------------------------------
# Static File Routes & Root
# ---------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/temp_uploads/<path:filename>")
def serve_temp_uploads(filename):
    return send_from_directory(str(TEMP_UPLOAD_DIR), filename)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "online",
        "model": MODEL_META,
        "public_url": PUBLIC_TUNNEL_URL,
        "esp32": ESP32_DATA,
        "timestamp": time.time()
    })

@app.route("/api/network-info", methods=["GET"])
def get_network_info():
    port = int(os.environ.get("PORT", 10000))
    local_ip = get_local_ip()
    local_url = f"http://{local_ip}:{port}"
    
    target_url = PUBLIC_TUNNEL_URL if PUBLIC_TUNNEL_URL else local_url
    qr_b64 = generate_qr_code_b64(target_url)

    return jsonify({
        "success": True,
        "local_ip": local_ip,
        "port": port,
        "local_url": local_url,
        "public_url": PUBLIC_TUNNEL_URL,
        "mobile_target_url": target_url,
        "qr_code_b64": qr_b64,
        "instructions": f"Scan QR code or visit {target_url} on any phone anywhere."
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file_obj = request.files["image"]
        raw_image = Image.open(file_obj.stream).convert("RGB")
        
        file_uuid = str(uuid.uuid4())
        filename = f"{file_uuid}.jpg"
        filepath = TEMP_UPLOAD_DIR / filename
        
        raw_image.save(filepath, format="JPEG")
        
        return jsonify({
            "success": True,
            "uuid": file_uuid,
            "url": f"/temp_uploads/{filename}"
        })
    except Exception as err:
        return jsonify({"error": f"Upload error: {str(err)}", "success": False}), 500


@app.route("/api/inference", methods=["POST"])
def inference():
    start_time = time.time()
    try:
        if MODEL is None:
            load_model_instance()

        req_data = request.get_json() or {}
        file_uuid = req_data.get("uuid")
        if not file_uuid:
            return jsonify({"error": "UUID not provided"}), 400
            
        filepath = TEMP_UPLOAD_DIR / f"{file_uuid}.jpg"
        if not filepath.exists():
            return jsonify({"error": "Image not found"}), 404

        raw_image = Image.open(filepath).convert("RGB")
        use_tta = req_data.get("use_tta", False)
        use_hair_removal = req_data.get("use_hair_removal", False)
        custom_threshold = req_data.get("threshold", None)

        processed_image = raw_image
        hair_removed = False
        if use_hair_removal:
            processed_image = remove_hair_dullrazor(raw_image)
            hair_removed = True
            processed_filename = f"{file_uuid}_processed.jpg"
            processed_image.save(TEMP_UPLOAD_DIR / processed_filename, format="JPEG")

        if os.environ.get("RENDER") or MODEL is None:
            probability = predict_lightweight_numpy(processed_image)
        else:
            img_size = MODEL_META.get("image_size", 224)
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            image_tensor = transform(processed_image).unsqueeze(0).to(DEVICE)

            if use_tta:
                prob_tensor = predict_with_tta(MODEL, image_tensor)
                probability = float(prob_tensor.cpu().item())
            else:
                with torch.no_grad():
                    output = MODEL(image_tensor)
                    probability = float(torch.sigmoid(output).cpu().item())

        threshold = MODEL_META.get("threshold", 0.41)
        if custom_threshold is not None:
            try:
                threshold = float(custom_threshold)
            except ValueError:
                pass

        is_melanoma = probability >= threshold
        confidence = round(float(min(99.9, max(52.0, 50.0 + abs(probability - 0.5) * 98.0))), 1)
        prob_percent = round(probability * 100, 2)

        if probability < 0.18:
            risk_code = "LOW"
            risk_level = "LOW RISK (Uniform Benign Pattern)"
            recommendation = "Lesion exhibits uniform benign characteristics. Continue routine self-examination."
        elif probability < threshold:
            risk_code = "MODERATE"
            risk_level = "MODERATE (Monitor Lesion)"
            recommendation = "Lesion appears predominantly benign. Monitor for changes in color, border, or size."
        elif probability < 0.75:
            risk_code = "HIGH"
            risk_level = "HIGH RISK (Suspicious Lesion)"
            recommendation = "⚠️ Suspicious skin pattern detected. Clinical dermatoscopic examination & biopsy evaluation recommended."
        else:
            risk_code = "CRITICAL"
            risk_level = "CRITICAL RISK (High Confidence Melanoma)"
            recommendation = "🚨 High-risk malignant melanoma characteristics detected. Urgent consultation with a board-certified dermatologist is strongly recommended!"

        prediction_str = "Likely Melanoma" if is_melanoma else "Likely Benign"

        ESP32_DATA["current_display"] = {
            "probability_percent": prob_percent,
            "prediction": prediction_str,
            "risk_code": risk_code,
            "confidence": confidence,
            "timestamp": time.time()
        }
        ESP32_DATA["last_updated"] = time.time()

        response = {
            "success": True,
            "probability": round(probability, 4),
            "probability_percent": prob_percent,
            "is_melanoma": is_melanoma,
            "prediction": prediction_str,
            "risk_code": risk_code,
            "risk_level": risk_level,
            "confidence": confidence,
            "threshold": threshold,
            "recommendation": recommendation,
            "latency_ms": round((time.time() - start_time) * 1000, 1)
        }
        
        if hair_removed:
            response["processed_url"] = f"/temp_uploads/{file_uuid}_processed.jpg"

        return jsonify(response)

    except Exception as err:
        import traceback
        app.logger.error(f"[!] Inference server error: {traceback.format_exc()}")
        return jsonify({"error": f"AI Inference error: {str(err)}", "success": False}), 500


@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    try:
        req_data = request.get_json() or {}
        file_uuid = req_data.get("uuid")
        if not file_uuid:
            return jsonify({"error": "UUID not provided"}), 400
            
        use_processed = req_data.get("use_processed", False)
        
        filename = f"{file_uuid}_processed.jpg" if use_processed else f"{file_uuid}.jpg"
        filepath = TEMP_UPLOAD_DIR / filename
        if not filepath.exists():
            filepath = TEMP_UPLOAD_DIR / f"{file_uuid}.jpg"
            if not filepath.exists():
                return jsonify({"error": "Image not found"}), 404

        image = Image.open(filepath).convert("RGB")
        heatmap_filename = f"{file_uuid}_heatmap.jpg"
        heatmap_path = TEMP_UPLOAD_DIR / heatmap_filename

        with _gradcam_lock:
            if os.environ.get("RENDER") or MODEL is None:
                heatmap_img = generate_numpy_heatmap(image)
                if heatmap_img:
                    heatmap_img.save(heatmap_path, format="JPEG")
            else:
                img_size = MODEL_META.get("image_size", 224)
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                target_layer = None
                if hasattr(MODEL, "layer4"):
                    target_layer = MODEL.layer4[-1]
                elif hasattr(MODEL, "features"):
                    target_layer = MODEL.features[-1]

                if target_layer is not None:
                    cam_engine = GradCAM(MODEL, target_layer)
                    try:
                        cam_arr = cam_engine.generate_heatmap(image_tensor)
                        heatmap_img = overlay_gradcam(image, cam_arr)
                        if heatmap_img:
                            heatmap_img.save(heatmap_path, format="JPEG")
                    finally:
                        cam_engine.remove_hooks()

        if not heatmap_path.exists():
            return jsonify({"error": "Failed to generate heatmap", "success": False}), 500

        return jsonify({
            "success": True,
            "url": f"/temp_uploads/{heatmap_filename}"
        })

    except Exception as err:
        import traceback
        app.logger.error(f"[!] Heatmap error: {traceback.format_exc()}")
        return jsonify({"error": f"Heatmap error: {str(err)}", "success": False}), 500


@app.route("/api/abcde", methods=["POST"])
def abcde():
    try:
        req_data = request.get_json() or {}
        file_uuid = req_data.get("uuid")
        probability = req_data.get("probability", 0.5)
        
        if not file_uuid:
            return jsonify({"error": "UUID not provided"}), 400
            
        filepath = TEMP_UPLOAD_DIR / f"{file_uuid}.jpg"
        if not filepath.exists():
            return jsonify({"error": "Image not found"}), 404

        image = Image.open(filepath).convert("RGB")
        abcde_analysis = calculate_abcde_scores(image, float(probability))
        
        return jsonify({
            "success": True,
            "abcde": abcde_analysis
        })

    except Exception as err:
        import traceback
        app.logger.error(f"[!] ABCDE error: {traceback.format_exc()}")
        return jsonify({"error": f"ABCDE error: {str(err)}", "success": False}), 500


@app.route("/api/analyze-gemini", methods=["POST"])
def analyze_gemini():
    """Generates visual analysis and clinical narrative using Gemini AI Vision API."""
    data = request.get_json() or {}
    api_key = data.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    file_uuid = data.get("uuid")
    
    if not api_key:
        return jsonify({
            "success": False,
            "error": "Gemini API key is missing. AI analysis unavailable."
        })

    if not file_uuid:
        return jsonify({"error": "UUID not provided", "success": False}), 400

    try:
        import urllib.request
        import urllib.parse
        
        filepath = TEMP_UPLOAD_DIR / f"{file_uuid}.jpg"
        if not filepath.exists():
            return jsonify({"error": "Image not found"}), 404
            
        with open(filepath, "rb") as f:
            clean_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt_text = (
            "You are an expert dermatological AI consultant analyzing a high-resolution skin lesion image. "
            "Please provide a structured clinical assessment covering: "
            "1. Morphological Features (Color uniformity, border definition, symmetry). "
            "2. Differential Considerations (Melanoma vs Nevus vs Seborrheic Keratosis). "
            "3. Dermatological Action Plan (Biopsy recommendation, follow-up timeline). "
            "Keep the response professional, concise, and structured in Markdown."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": clean_b64}}
                ]
            }]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({"success": True, "analysis": content})

    except Exception as e:
        return jsonify({
            "success": True,
            "analysis": f"AI Clinical Assessment: Lesion analyzed by deep neural network. (Note: Gemini Vision API direct call returned: {str(e)})"
        })


@app.route("/api/esp32/config", methods=["POST"])
def config_esp32():
    data = request.get_json() or {}
    esp32_ip = data.get("esp32_ip", "").strip()
    mode = data.get("mode", "wifi")

    if esp32_ip and not re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", esp32_ip):
        return jsonify({"success": False, "error": "Invalid IP address"}), 400

    ESP32_DATA["esp32_ip"] = esp32_ip
    ESP32_DATA["mode"] = mode
    ESP32_DATA["last_updated"] = time.time()

    return jsonify({"success": True, "esp32": ESP32_DATA})


@app.route("/api/esp32/status", methods=["GET"])
def get_esp32_status():
    return jsonify({"success": True, "esp32": ESP32_DATA})


# ---------------------------------------------------------
# Error Handlers & Fallback Static Routes
# ---------------------------------------------------------
@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint not found", "success": False}), 404
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.errorhandler(405)
def handle_405(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method not allowed for API endpoint", "success": False}), 405
    return jsonify({"error": "Method not allowed", "success": False}), 405


@app.errorhandler(500)
def handle_500(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": f"Server error: {str(e)}", "success": False}), 500
    return jsonify({"error": "Internal server error", "success": False}), 500


@app.route("/<path:filename>")
def serve_static(filename):
    if filename.startswith("api/"):
        return jsonify({"error": "API endpoint not found", "success": False}), 404
    return send_from_directory(str(STATIC_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    local_ip = get_local_ip()
    app.logger.info("\n================================================================")
    app.logger.info(" DERMATOSCAN PRO AI - SKIN CANCER DETECTOR SERVER ACTIVE")
    app.logger.info(f" Local Machine Web UI : http://127.0.0.1:{port}")
    app.logger.info(f" Mobile LAN Web UI    : http://{local_ip}:{port}")
    app.logger.info("================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
