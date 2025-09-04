# streamlit_app.py
import importlib
import traceback
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, VideoHTMLAttributes

st.set_page_config(page_title="Realtime Inference Web App", layout="wide")
st.title("Realtime Inference — drop your inference_tflite.py in this folder")

# Try to import user's inference file dynamically.
MODEL_MODULE_NAME = "inference_tflite"  # name of your file (without .py)

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
    st.sidebar.text(str(model_mod))
else:
    st.sidebar.warning(f"`{MODEL_MODULE_NAME}.py` not found or import failed. Place it here and refresh.")

# settings
FRAME_RATE = st.sidebar.slider("Process FPS (approx)", min_value=1, max_value=30, value=8)
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.4)

class InferenceTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0
        self.model = model_mod
        # discover prediction API
        self.predict_fn = None
        if self.model:
            # Common patterns:
            # 1) module.predict(frame) -> result
            # 2) module.Inference() with instance.predict(frame)
            # 3) module.infer(frame)
            if hasattr(self.model, "predict") and callable(self.model.predict):
                self.predict_fn = lambda f: self.model.predict(f)
            elif hasattr(self.model, "infer") and callable(self.model.infer):
                self.predict_fn = lambda f: self.model.infer(f)
            elif hasattr(self.model, "Inference"):
                try:
                    inst = self.model.Inference()
                    if hasattr(inst, "predict"):
                        self.predict_fn = lambda f: inst.predict(f)
                    elif hasattr(inst, "infer"):
                        self.predict_fn = lambda f: inst.infer(f)
                except Exception:
                    # fallback: try class constructor with no args failed
                    st.sidebar.error("Failed to instantiate Inference() from your module.")
            else:
                st.sidebar.warning("No recognized `predict`/`infer` function or `Inference` class found in module.")

        # small smoothing buffer
        self.last_label = None
        self.buffer_text = ""

    def transform(self, frame):
        # frame: av.VideoFrame
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        # throttle processing to FRAME_RATE
        self.frame_count += 1
        if now - self.last_time < (1.0 / FRAME_RATE):
            # just return with overlay of last result
            self._draw_overlay(img, self.last_label)
            return img
        self.last_time = now

        # run model if available
        result_text = None
        try:
            if self.predict_fn is not None:
                # Model may expect RGB, BGR, or resized input — adapt inside your inference_tflite.py
                # We'll pass the BGR numpy array; adapt your module to accept this.
                pred = self.predict_fn(img)
                # Standardize result to string or dict with 'label' and optional 'score'
                if isinstance(pred, str):
                    result_text = pred
                elif isinstance(pred, dict):
                    # prefer label or text keys
                    result_text = pred.get("label") or pred.get("text") or str(pred)
                else:
                    # anything else -> string
                    result_text = str(pred)
            else:
                result_text = "no predict() found in module"
        except Exception as e:
            result_text = f"model error: {e}"
            st.sidebar.text(traceback.format_exc())

        self.last_label = result_text
        # optional: accumulate into buffer if it's a token
        if result_text and len(result_text) < 50:
            # heuristics to append (customize)
            self.buffer_text += result_text + " "

        self._draw_overlay(img, result_text)
        return img

    def _draw_overlay(self, img: Any, text: str):
        h, w = img.shape[:2]
        # semi-transparent rectangle
        cv2.rectangle(img, (0, h - 50), (w, h), (0, 0, 0), -1)
        display_text = text or ""
        cv2.putText(img, display_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

st.sidebar.markdown("### Controls")
if st.sidebar.button("Reload model module"):
    # trigger reload
    try:
        import importlib, sys
        if MODEL_MODULE_NAME in sys.modules:
            del sys.modules[MODEL_MODULE_NAME]
    except Exception:
        pass
    st.experimental_rerun()

RTC_VIDEO_HTML = VideoHTMLAttributes(autoPlay=True, controls=False, playsInline=True)

webrtc_ctx = webrtc_streamer(
    key="realtime-inference",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=InferenceTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs=RTC_VIDEO_HTML,
)

# show buffer and latest result
st.sidebar.markdown("### Live output")
if webrtc_ctx.video_transformer:
    transformer = webrtc_ctx.video_transformer
    last_text = getattr(transformer, "last_label", "")
    st.sidebar.text_area("Last prediction", value=last_text, height=80)
    st.sidebar.text_area("Buffer (accumulated short tokens)", value=getattr(transformer, "buffer_text", ""), height=120)
else:
    st.sidebar.info("Video transformer not initialized yet. Start camera in main area.")

st.markdown("""
**Notes**
- Place your `inference_tflite.py` in the same folder and make sure it exposes either:
  - `predict(frame: np.ndarray) -> str|dict` OR
  - `infer(frame: np.ndarray) -> ...` OR
  - `class Inference: def predict(self, frame): ...`
- The app passes BGR `numpy.ndarray` frames from OpenCV. Convert to RGB/resized inside your module as needed.
""")
