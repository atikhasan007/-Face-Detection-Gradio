import gradio as gr
import cv2
import numpy as np
from PIL import Image

# --- Load the Haarcascade classifier ---
HAARCASCADE_PATH = "haarcascade_frontalface_default.xml"
try:
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load Haarcascade classifier from {HAARCASCADE_PATH}")
    print("Haarcascade classifier loaded successfully!")
except Exception as e:
    print(f"Error loading Haarcascade classifier: {e}")
    face_cascade = None


# --- Face Detection Function ---
def detect_faces_from_webcam(image: Image.Image, scale_factor: float = 1.1, min_neighbors: int = 5):
    if image is None:
        return None
    if face_cascade is None:
        print("Haarcascade classifier not loaded.")
        return image

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    annotated_image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return annotated_image_pil


# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎥 Real-time Haarcascade Face Detection  
        Stream your webcam to detect faces in real time using OpenCV Haarcascade.
        """
    )

    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(
                type="pil",
                label="Your Webcam Feed",
                sources=["webcam"],
                streaming=True
            )

            scale_factor_slider = gr.Slider(
                minimum=1.01,
                maximum=1.5,
                value=1.1,
                step=0.01,
                label="Scale Factor (Detection Sensitivity)"
            )

            min_neighbors_slider = gr.Slider(
                minimum=0,
                maximum=10,
                value=5,
                step=1,
                label="Min Neighbors (Detection Quality)"
            )

        with gr.Column():
            output_image = gr.Image(type="pil", label="Detected Faces (Live Output)")

    webcam_input.stream(
        fn=detect_faces_from_webcam,
        inputs=[webcam_input, scale_factor_slider, min_neighbors_slider],
        outputs=output_image
    )

demo.launch()
