from dash import Dash, dcc, html, Input, Output, State, callback
import datetime, threading
from bird_species_data import class_names
from ultralytics import YOLO
import tensorflow as tf

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


model_loading_thread = threading.Thread(target=load_model)
model_loading_thread.start()

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


def parse_contents(contents):
    confidence = 0.9
    return html.Div(
        className="max-w-sm rounded overflow-hidden shadow-lg",
        children=[
            html.Img(
                src=contents,
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
                                children=f"{confidence:.2%}",
                                style={"width": f"{confidence:.2%}"},
                            )
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
        children = [parse_contents(c) for c in list_of_contents]
        return children


if __name__ == "__main__":
    app.run(debug=True)
