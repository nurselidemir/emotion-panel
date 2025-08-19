import io
import os
import tempfile
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
import cv2
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from collections import Counter

TF_LOADED = False
tf = None
keras = None

from config import Config  
from model_irse import Backbone


MODELS_DIR = "models"
WAV2VEC2_DIR = os.path.join(MODELS_DIR, "wac2vec2")


MODEL_FILES = {
    "googlenet": os.path.join(MODELS_DIR, "googleNet_model.pth"),
    "resnet18_video": os.path.join(MODELS_DIR, "resnet18_ravdess.pth"),
    "efficientfer": os.path.join(MODELS_DIR, "efficientnetv2b0_fer2013plus_augmented.h5"),
    "ferformer": os.path.join(MODELS_DIR, "ferformer_model.pt"),
    "fast_rcnn": os.path.join(MODELS_DIR, "emotion_alexnet_model.pth"),
    "mobilevit": os.path.join(MODELS_DIR, "mobilevit_fer2013plus.pth"),
}


HF_REPOS = {
    "googlenet":      "nurselidemir/emotion-googlenet-fer2013plus",
    "fast_rcnn":      "nurselidemir/emotion-alexnet-fast-rcnn-fer2013plus",
    "ferformer":      "nurselidemir/emotion-ferformer-fer2013plus",
    "efficientfer":   "nurselidemir/emotion-efficientfer-keras",
    "resnet18_video": "nurselidemir/emotion-resnet18-ravdess-video",
    "mobilevit":      "nurselidemir/emotion-mobilevit-fer2013plus",
    "wav2vec2":       "nurselidemir/emotion-wav2vec2-ravdess-audio",
}


MODEL_PERFORMANCE = {
    "googlenet":      {"dataset": "Fer2013Plus",           "training_time": 2385.17, "test_time": 44.08,  "accuracy": 81.84, "resource": "T4"},
    "fast_rcnn":      {"dataset": "Fer2013Plus",           "training_time": 1380.10, "test_time": 277.50, "accuracy": 81.11, "resource": "T4"},
    "ferformer":      {"dataset": "Fer2013Plus",           "training_time": 10246.47,"test_time": 83.15,  "accuracy": 72.47, "resource": "T4"},
    "efficientfer":   {"dataset": "Fer2013Plus",           "training_time": 6671.73, "test_time": 28.29,  "accuracy": 78.94, "resource": "T4"},
    "resnet18_video": {"dataset": "RAVDESS speech video",  "training_time": 613.99,  "test_time": 779.73, "accuracy": 91.26, "resource": "T4"},
    "wav2vec2":       {"dataset": "RAVDESS speech audio",  "training_time": 1776.22, "test_time": 15.43,  "accuracy": 86.81, "resource": "T4"},
    "mobilevit":      {"dataset": "Fer2013Plus",           "training_time": 2714.53, "test_time": 327.67, "accuracy": 81.76, "resource": "T4"},
}

MODEL_CATEGORIES = {
    "image": ["googlenet", "efficientfer", "ferformer", "fast_rcnn", "mobilevit"],
    "video": ["resnet18_video"],
    "audio": ["wav2vec2"],
}


CLASS_NAMES_8 = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
CLASS_NAMES_8_VIDEO = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
CLASS_NAMES_8_IMG = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
CLASS_NAMES_8_EFF = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'contempt']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_loaded: Dict[str, object] = {}
_processor_w2v2 = None
_id2label_w2v2: Optional[Dict[int, str]] = None



def _resolve_weight(local_path: str, hf_repo: str, filename: Optional[str] = None) -> str:
    """
    1) local_path varsa onu kullanır
    2) yoksa HF Hub'dan indirip dosya/dizin yolunu döndürür
    """
    import pathlib
    p = pathlib.Path(local_path)
    use_hf_only = os.getenv("USE_HF_ONLY", "0") == "1"

    if not use_hf_only and p.exists() and (p.is_file() or (p.is_dir() and any(p.iterdir()))):
        return str(p if (p.is_file() or filename is None) else p / filename)

    
    try:
        if filename:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=hf_repo, filename=filename)
        else:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=hf_repo)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model dosyası indirilemedi: {hf_repo}/{filename or ''} ({e})")


def _ensure_tensorflow():
    global TF_LOADED, tf, keras
    if TF_LOADED:
        return
    import tensorflow as tf_mod
    from tensorflow import keras as keras_mod
    TF_LOADED = True
    tf = tf_mod
    keras = keras_mod


def _read_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _img_transform_224():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _img_to_tensor(img: Image.Image):
    return _img_transform_224()(img).unsqueeze(0).to(device)


def load_wav2vec2():
    global _processor_w2v2, _id2label_w2v2
    if "wav2vec2" in _loaded:
        return _loaded["wav2vec2"]

   
    local_or_hf_dir = _resolve_weight(
        local_path=WAV2VEC2_DIR,
        hf_repo=HF_REPOS["wav2vec2"],
        filename=None
    )

    from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
    _processor_w2v2 = Wav2Vec2FeatureExtractor.from_pretrained(local_or_hf_dir)
    model = AutoModelForAudioClassification.from_pretrained(local_or_hf_dir)
    model.to(device).eval()

    _id2label_w2v2 = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()} or {
        i: c for i, c in enumerate(CLASS_NAMES_8)
    }
    _loaded["wav2vec2"] = model
    return model


def load_resnet18_video():
    if "resnet18_video" in _loaded:
        return _loaded["resnet18_video"]
    weight_path = _resolve_weight(MODEL_FILES["resnet18_video"], HF_REPOS["resnet18_video"], "resnet18_ravdess.pth")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    _loaded["resnet18_video"] = model
    return model


def load_googlenet():
    if "googlenet" in _loaded:
        return _loaded["googlenet"]
    weight_path = _resolve_weight(MODEL_FILES["googlenet"], HF_REPOS["googlenet"], "googleNet_model.pth")
    model = models.googlenet(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 8)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    _loaded["googlenet"] = model
    return model


def load_efficientfer():
    if "efficientfer" in _loaded:
        return _loaded["efficientfer"]
    _ensure_tensorflow()
    weight_path = _resolve_weight(MODEL_FILES["efficientfer"], HF_REPOS["efficientfer"], "efficientnetv2b0_fer2013plus_augmented.h5")
    model = keras.models.load_model(weight_path)
    _loaded["efficientfer"] = model
    return model


def load_ferformer():
    if "ferformer" in _loaded:
        return _loaded["ferformer"]
    weight_path = _resolve_weight(MODEL_FILES["ferformer"], HF_REPOS["ferformer"], "ferformer_model.pt")
    input_side = 112
    backbone = Backbone(input_size=(input_side, input_side), num_layers=50).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, 3, input_side, input_side).to(device)
        feat_dim = backbone(dummy).view(1, -1).shape[1]
    model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(feat_dim, 8)).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    _loaded["ferformer"] = model
    return model


def load_fast_rcnn():
    if "fast_rcnn" in _loaded:
        return _loaded["fast_rcnn"]
    weight_path = _resolve_weight(MODEL_FILES["fast_rcnn"], HF_REPOS["fast_rcnn"], "emotion_alexnet_model.pth")
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 8)
    state = torch.load(weight_path, map_location=device)
    if any(k.startswith("backbone.") for k in state.keys()):
        state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    _loaded["fast_rcnn"] = model
    return model


def load_mobilevit():
    if "mobilevit" in _loaded:
        return _loaded["mobilevit"]
    weight_path = _resolve_weight(MODEL_FILES["mobilevit"], HF_REPOS["mobilevit"], "mobilevit_fer2013plus.pth")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    _loaded["mobilevit"] = model
    return model


def predict_audio_wav2vec2(wav_bytes: bytes):
    model = load_wav2vec2()
    speech, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    inputs = _processor_w2v2(speech.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = int(torch.argmax(logits, dim=-1).item())
    return _id2label_w2v2.get(predicted_id, CLASS_NAMES_8[predicted_id] if predicted_id < len(CLASS_NAMES_8) else str(predicted_id))


def predict_video_resnet18(mp4_bytes: bytes, sample_count: int = 10):
    model = load_resnet18_video()


    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp_file.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(mp4_bytes)

        cap = cv2.VideoCapture(tmp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise HTTPException(status_code=400, detail="Video okunamadı.")

        sample_count = max(1, min(sample_count, frame_count))
        frame_indices = [int(i * frame_count / sample_count) for i in range(sample_count)]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        predictions = []
        frame_id = 0
        success = True
        while success and frame_id <= max(frame_indices):
            success, frame = cap.read()
            if frame_id in frame_indices and success:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                x = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                    _, pred = torch.max(out, 1)
                    predictions.append(pred.item())
            frame_id += 1
        cap.release()

        if not predictions:
            raise HTTPException(status_code=400, detail="Videodan frame alınamadı.")

        majority = Counter(predictions).most_common(1)[0][0]
        return CLASS_NAMES_8_VIDEO[majority]
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def predict_image_pytorch(img: Image.Image, torch_model: torch.nn.Module, class_names):
    x = _img_to_tensor(img)
    with torch.no_grad():
        out = torch_model(x)
        pred = out.argmax(1).item()
    return class_names[pred]


def predict_image_efficientfer(img: Image.Image):
    _ensure_tensorflow()
    model = load_efficientfer()
    arr = keras.preprocessing.image.img_to_array(img.resize((224, 224))) / 255.0
    arr = arr[None, ...]
    preds = model.predict(arr)
    pred_idx = int(preds.argmax(axis=1)[0])
    return CLASS_NAMES_8_EFF[pred_idx] if pred_idx < len(CLASS_NAMES_8_EFF) else str(pred_idx)


def predict_image_ferformer(img: Image.Image):
    model = load_ferformer()
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
    return CLASS_NAMES_8_IMG[pred]


def predict_image_fast_rcnn(img: Image.Image):
    model = load_fast_rcnn()
    x = _img_to_tensor(img)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
    return CLASS_NAMES_8_IMG[pred]


def predict_image_mobilevit(img: Image.Image):
    model = load_mobilevit()
    x = _img_to_tensor(img)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
    return CLASS_NAMES_8_IMG[pred]


class PredictResponse(BaseModel):
    model_name: str
    predicted_emotion: str
    model_config = {"protected_namespaces": ()}


app = FastAPI(title="Emotion Panel API", version="1.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Emotion Analysis Panel API",
        "version": "1.4.0",
        "status": "running"
    }


@app.get("/models")
def list_models():
    def to_entry(name: str):
        acc = MODEL_PERFORMANCE.get(name, {}).get("accuracy")
        return {"name": name, "accuracy": f"{acc:.2f}%" if isinstance(acc, (int, float)) else None}
    return {
        "image": [to_entry(m) for m in MODEL_CATEGORIES["image"]],
        "video": [to_entry(m) for m in MODEL_CATEGORIES["video"]],
        "audio": [to_entry(m) for m in MODEL_CATEGORIES["audio"]],
    }


@app.get("/models/performance")
def get_model_performance():
    return MODEL_PERFORMANCE


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), model_name: str = Form(...), category: str = Form(...)):
    category = category.lower()
    model_name = model_name.strip().lower()

    allowed_exts = {
        "image": [".png"],
        "video": [".mp4"],
        "audio": [".wav"]
    }
    _, ext = os.path.splitext(file.filename.lower())
    if category not in allowed_exts or ext not in allowed_exts[category]:
        raise HTTPException(
            status_code=400,
            detail=f"{category} için yalnızca şu formatlar destekleniyor: {', '.join(allowed_exts.get(category, []))}"
        )

    if category == "audio" and model_name == "wav2vec2":
        return PredictResponse(model_name=model_name, predicted_emotion=predict_audio_wav2vec2(await file.read()))
    elif category == "video" and model_name == "resnet18_video":
        return PredictResponse(model_name=model_name, predicted_emotion=predict_video_resnet18(await file.read()))
    elif category == "image":
        data = await file.read()
        img = _read_image_from_bytes(data)
        if model_name == "googlenet":
            return PredictResponse(model_name=model_name, predicted_emotion=predict_image_pytorch(img, load_googlenet(), CLASS_NAMES_8_IMG))
        elif model_name == "efficientfer":
            return PredictResponse(model_name=model_name, predicted_emotion=predict_image_efficientfer(img))
        elif model_name == "ferformer":
            return PredictResponse(model_name=model_name, predicted_emotion=predict_image_ferformer(img))
        elif model_name == "fast_rcnn":
            return PredictResponse(model_name=model_name, predicted_emotion=predict_image_fast_rcnn(img))
        elif model_name == "mobilevit":
            return PredictResponse(model_name=model_name, predicted_emotion=predict_image_mobilevit(img))

    raise HTTPException(status_code=400, detail=f"{model_name} bu kategori için geçerli değil.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
