# streamlit_app.py
import importlib
import traceback
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, VideoHTMLAttributes

st.set_page_config(page_title="GestureSpeak — Morse Code", layout="wide")
st.title("GestureSpeak — Realtime Morse Code Inference")

# -----------------------------
# Dynamic model import
# -----------------------------
MODEL_MODULE_NAME = "inference_tflite"

def load_model_module():
    try:
        mod = importlib.import_module(MODEL_MODULE_NAME)
        importlib.reload(mod)
        return mod
    except Exception as e:
        st.sidebar.error(f"Failed to import `{MODEL_MODULE_NAME}.py`:\n{e}")
        st.sidebar.text(traceback.format_exc())
        return None

model_mod = load_model_module()

st.sidebar.markdown("### Model module status")
if model_mod:
    st.sidebar.success(f"Imported `{MODEL_MODULE_NAME}.py`")
else:
    st.sidebar.warning(f"`{MODEL_MODULE_NAME}.py` not found or import failed. Place it here and refresh.")

# -----------------------------
# Settings
# -----------------------------
FRAME_RATE = st.sidebar.slider("Process FPS (approx)", min_value=1, max_value=30, value=8)

# -----------------------------
# Video transformer
# -----------------------------
class InferenceTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0
        self.model = model_mod
        self.predict_fn = None
        if self.model and hasattr(self.model, "predict") and callable(self.model.predict):
            self.predict_fn = lambda f: self.model.predict(f)
        self.last_label = None
        self.buffer_text = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        self.frame_count += 1

        if now - self.last_time < (1.0 / FRAME_RATE):
            self._draw_overlay(img, self.last_label)
            return img
        self.last_time = now

        result_text = None
        try:
            if self.predict_fn is not None:
                pred = self.predict_fn(img)
                if isinstance(pred, str):
                    result_text = pred
                elif isinstance(pred, dict):
                    result_text = pred.get("text") or str(pred)
                else:
                    result_text = str(pred)
            else:
                result_text = "no predict() found in module"
        except Exception as e:
            result_text = f"model error: {e}"
            st.sidebar.text(traceback.format_exc())

        self.last_label = result_text

        if result_text and len(str(result_text)) < 50:
            self.buffer_text += str(result_text) + " "

        self._draw_overlay(img, result_text)
        return img

    def _draw_overlay(self, img: Any, text: str):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 50), (w, h), (0, 0, 0), -1)
        display_text = text or ""
        cv2.putText(img, display_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

# -----------------------------
# Controls
# -----------------------------
st.sidebar.markdown("### Controls")
if st.sidebar.button("Reload model module"):
    try:
        import sys
        if MODEL_MODULE_NAME in sys.modules:
            del sys.modules[MODEL_MODULE_NAME]
    except Exception:
        pass
    st.experimental_rerun()

RTC_VIDEO_HTML = VideoHTMLAttributes(autoPlay=True, controls=False, playsInline=True)

webrtc_ctx = webrtc_streamer(
    key="realtime-inference",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=InferenceTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs=RTC_VIDEO_HTML,
)

# -----------------------------
# Live output
# -----------------------------
st.sidebar.markdown("### Live output")
if webrtc_ctx.video_transformer:
    transformer = webrtc_ctx.video_transformer
    last_text = getattr(transformer, "last_label", "")
    st.sidebar.text_area("Last prediction", value=last_text, height=80)
    st.sidebar.text_area("Buffer (accumulated short tokens)", value=getattr(transformer, "buffer_text", ""), height=120)
else:
    st.sidebar.info("Video transformer not initialized yet. Start camera in main area.")
