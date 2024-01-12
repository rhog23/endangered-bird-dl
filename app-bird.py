from dash import Dash, dcc, html, Input, Output, State, callback
import cv2, threading, base64, io
from bird_species_data import class_names
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np

external_scripts = [
    "https://tailwindcss.com/",
    {"src": "https://cdn.tailwindcss.com"},
]


# Load TFlite model in the background thread
def load_model():
    global det_model
    global cls_model
    global input_details
    global output_details

    det_model = YOLO("assets/yolov8s-lite/yolov8s_float16.tflite", task="detect")
    cls_model = tf.lite.Interpreter(model_path="assets/models/endb-cls-mb3.tflite")
    cls_model.allocate_tensors()

    input_details = cls_model.get_input_details()
    output_details = cls_model.get_output_details()

    print("[INFO] Model sucessfully loaded ✅")


def detect_bird(c):
    results = []
    decoded_bytes = base64.b64decode(c.split(",")[1])
    image = Image.open(io.BytesIO(decoded_bytes))
    image_array = np.array(image)
    det_result = det_model.predict(image, imgsz=192, classes=14, conf=0.5)
    for result in det_result:
        for box in result.boxes:
            box = box.xyxy
            x = int(box[0][0])
            y = int(box[0][1])
            w = int(box[0][2])
            h = int(box[0][3])

            crop = image_array[y:h, x:w]
            bird_species, conf = classify_bird(crop)
            results.append(zip(crop, bird_species, conf))

    return results


def classify_bird(image):
    input_shape = input_details[0]["shape"]
    input_data = np.expand_dims(
        cv2.cvtColor(
            cv2.resize(image, (input_shape[1], input_shape[2])), cv2.COLOR_BGR2RGB
        ),
        axis=0,
    ).astype(np.float32)
    cls_model.set_tensor(input_details[0]["index"], input_data)
    cls_model.invoke()
    output_data = cls_model.get_tensor(output_details[0]["index"])
    bird_species = class_names[np.argmax(output_data)]
    conf = np.max(output_data)

    return bird_species, conf


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


def parse_contents(img, result):
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
                        className="font-medium text-slate-900 text-sm mb-1 ",
                        children="Result",
                    ),
                    html.Div(
                        className="w-full bg-gray-200 rounded-full",
                        children=[
                            html.Div(
                                className="bg-green-400 text-xs font-medium text-center p-0.5 leading-none rounded-full number",
                                children=f"{label}, {conf:.2%}",
                                style={"width": f"{conf:.2%}"},
                            )
                            for crop, label, conf in result
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
    list_of_results = []
    if list_of_contents is not None:
        for c in list_of_contents:
            image_result = detect_bird(c)
            list_of_results.append(zip(c, image_result))
        children = [parse_contents(img, result) for img, result in list_of_results]
        return children


if __name__ == "__main__":
    model_loading_thread = threading.Thread(target=load_model)
    model_loading_thread.start()
    app.run(debug=True)
