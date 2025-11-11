
import gradio as gr
import cv2
import numpy as np
from PIL import Image

# Load the Haarcascade classifier ---
# Make sure 'haarcascade_frontalface_default.xml' is in the same directory as this app.py
HAARCASCADE_PATH = "/home/atik/Desktop/model_deployment/haar_cascade.xml"
try:
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load Haarcascade classifier from {HAARCASCADE_PATH}")
    print("Haarcascade classifier loaded successfully!")
except Exception as e:
    print(f"Error loading Haarcascade classifier: {e}")
    # You might want to add a fallback or an error message in the UI if it fails.
    face_cascade = None # Set to None so the detection function can handle it.


# Face Detection Function ---
def detect_faces_from_webcam(image: Image.Image, scale_factor: float = 1.1, min_neighbors: int = 5):
    if image is None:
        return None # Return None if no image is provided (e.g., webcam not active yet)
    if face_cascade is None:
        # If cascade failed to load, return the original image with an error message
        # (This is a simple way to handle, could be more sophisticated)
        print("Haarcascade classifier not loaded. Cannot perform detection.")
        return image # Or draw text on image: cv2.putText(np.array(image), "Error: Classifier not loaded", ...)

    # Convert PIL Image to OpenCV format (BGR)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # OpenCV works in BGR by default
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30) # Minimum object size to be detected. Objects smaller than this are ignored.
    )

    # Draw rectangles around the faces on the BGR image
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle

    # Convert back to PIL Image (RGB) for Gradio display
    annotated_image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return annotated_image_pil

# --- 3. Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Real-time Haarcascade Face Detection
        Stream your front camera to detect faces using the OpenCV Haarcascade classifier.
        """
    )
    with gr.Row():
        with gr.Column():
            # Webcam input for continuous streaming
            webcam_input = gr.Image(
                type="pil",
                label="Your Webcam Feed",
                sources=["webcam"], # Enable webcam as input source
                streaming=True,     # Enable real-time streaming
                # mirror_webcam=True # Optionally mirror the webcam feed
            )
            # Sliders to adjust Haarcascade parameters
            scale_factor_slider = gr.Slider(
                minimum=1.01,
                maximum=1.5,
                value=1.1,
                step=0.01,
                label="Scale Factor (detection sensitivity)"
            )
            min_neighbors_slider = gr.Slider(
                minimum=0,
                maximum=10,
                value=5,
                step=1,
                label="Min Neighbors (detection quality)"
            )

        with gr.Column():
            output_image = gr.Image(
                type="pil",
                label="Detected Faces (Real-time)"
            )

    # Stream the input image to the detection function
    webcam_input.stream(
        fn=detect_faces_from_webcam,
        inputs=[webcam_input, scale_factor_slider, min_neighbors_slider],
        outputs=output_image,
        stream_every=0.1, # Process a frame every 0.1 seconds (10 FPS)
        time_limit=300 # Limit stream to 5 minutes to manage resource usage
    )

# demo.launch()
demo.launch(share = True)

