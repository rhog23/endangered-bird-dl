from dash import Dash, dcc, html, Input, Output, callback
import cv2, threading, base64, io
from matplotlib.pyplot import cla
from new_classes import class_names
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
import keras_cv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable GPU usage to avoid CUDA/cuDNN issues
tf.config.set_visible_devices([], "GPU")
logger.info(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

# Disable XLA JIT compilation to avoid JIT-related errors
tf.config.optimizer.set_jit(False)

external_scripts = [
    "https://tailwindcss.com/",
    {"src": "https://cdn.tailwindcss.com"},
]


# Load Keras model in the background thread
def load_model():
    global det_model
    global cls_model

    det_model = YOLO("yolo11s.pt", task="detect")
    # Provide custom objects to handle PatchingAndEmbedding layer
    # custom_objects = {"PatchingAndEmbedding": keras_cv.layers.PatchingAndEmbedding}
    # cls_model = tf.keras.models.load_model(
    #     "assets/models/eb-tinyvit-val-fine.keras", custom_objects=custom_objects
    # )
    cls_model = tf.keras.models.load_model("assets/models/eb-mb3-acc-fine.keras")

    print("[INFO] Model successfully loaded ✅")


def detect_bird(c):
    results = []
    decoded_bytes = base64.b64decode(c.split(",")[1])
    image = Image.open(io.BytesIO(decoded_bytes))
    image_array = np.array(image)
    det_result = det_model.predict(image, classes=14, conf=0.3, device=0)
    for result in det_result:
        for box in result.boxes:
            box = box.xyxy
            x = int(box[0][0])
            y = int(box[0][1])
            w = int(box[0][2])
            h = int(box[0][3])

            crop = image_array[y:h, x:w]
            _, img_encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            bird_species, conf = classify_bird(crop)
            results.append(
                {
                    "cropped_image": base64.b64encode(img_encoded).decode("utf-8"),
                    "label": bird_species,
                    "confidence": conf,
                }
            )

    return results


def classify_bird(image):
    try:
        # Assuming the model expects input shape (height, width, channels)
        input_shape = cls_model.input_shape[1:3]  # Get height and width
        logger.info(f"Model expected input shape: {input_shape}")

        # Resize image
        resized_image = cv2.resize(image, input_shape)
        logger.info(f"Resized image shape: {resized_image.shape}")

        # Convert to float32 and normalize
        input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
        # input_data = input_data / 255.0

        # Check for invalid values
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            logger.error("Input data contains NaN or Inf values")
            raise ValueError("Invalid input data: contains NaN or Inf values")

        logger.info(f"Input data shape: {input_data.shape}")

        # Get model prediction
        output_data = cls_model.predict(input_data, verbose=0)
        bird_species = class_names[np.argmax(output_data[0])]
        conf = np.max(output_data[0])

        return bird_species, conf
    except Exception as e:
        logger.error(f"Error in classify_bird: {str(e)}")
        raise


app = Dash(__name__, external_scripts=external_scripts)

app.layout = html.Div(
    [
        html.Section(
            className="w-full flex flex-center flex-col gap-3",
            children=[
                html.H1(
                    "Endangered Bird Identifier App",
                    className="text-5xl custom-header-font mt-3 self-center",
                ),
                dcc.Upload(
                    id="upload-image",
                    className="border-dashed border border-sky-500 p-8 mt-16 max-w-5xl mx-auto rounded-md",
                    children=html.Div(
                        [
                            html.P("Drag and Drop or "),
                            html.A("Select Files", className="font-bold"),
                        ],
                        className="flex justify-center gap-2 text-2xl",
                    ),
                    # Allow multiple files to be uploaded
                    multiple=True,
                ),
                html.Div(
                    id="output-image-upload",
                    className="grid grid-cols-5 gap-4 self-center",
                ),
            ],
        )
    ]
)


def parse_contents(img, results):
    return html.Div(
        className="max-w-sm rounded overflow-hidden shadow-lg",
        children=[
            html.Img(
                src=img,
                className="w-full object-scale-down min-h-64 max-h-64 bg-slate-900",
            ),
            # label and confidence bar
            html.Div(
                className="px-6 py-4",
                children=[
                    html.Div(
                        className="font-medium text-slate-900 mb-1 ",
                        children="Identification Result",
                    ),
                    # cropped image, label, and confidence section
                    html.Div(
                        className="overflow-auto min-h-48 relative max-w-sm mx-auto bg-white flex flex-col divide-y",
                        children=[
                            html.Div(
                                className="flex items-center gap-4 p-4 shadow-lg",
                                children=[
                                    html.Img(
                                        src=f"data:image/png;base64,{result['cropped_image']}",
                                        className="min-w-16 h-16 rounded-full object-scale-down bg-slate-900",
                                    ),
                                    html.Div(
                                        className="w-full flex flex-col",
                                        children=[
                                            html.P(
                                                className="font-medium text-sm text-slate-900",
                                                children=f"{result['label']}",
                                            ),
                                            html.Span(
                                                className="bg-green-400 text-xs font-medium text-center p-0.5 leading-none rounded-full number",
                                                children=f"{result['confidence']:.2%}",
                                                style={
                                                    "width": f"{result['confidence']:.2%}"
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                            )
                            for result in results
                        ],
                    ),
                ],
            ),
        ],
    )


@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
)
def update_output(list_of_contents):
    if list_of_contents is not None:
        children = [parse_contents(c, detect_bird(c)) for c in list_of_contents]
        return children


if __name__ == "__main__":
    model_loading_thread = threading.Thread(target=load_model)
    model_loading_thread.start()
    app.run(debug=True)
