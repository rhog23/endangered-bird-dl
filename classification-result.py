import cv2
import threading
import sys
from new_classes import class_names
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import keras_cv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable GPU usage to avoid CUDA/cuDNN issues
tf.config.set_visible_devices([], "GPU")
logger.info(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

# Disable XLA JIT compilation
tf.config.optimizer.set_jit(False)

# Global model variables
det_model = None
cls_model = None


# Load models in a background thread
def load_model():
    global det_model, cls_model
    try:
        det_model = YOLO("yolo11s.pt", task="detect")
        custom_objects = {"PatchingAndEmbedding": keras_cv.layers.PatchingAndEmbedding}
        cls_model = tf.keras.models.load_model(
            "assets/models/eb-tinyvit-val-fine.keras", custom_objects=custom_objects
        )
        logger.info("Models successfully loaded ✅")
        logger.info(f"YOLO class names: {det_model.names}")
        logger.info(f"Classification model input shape: {cls_model.input_shape}")
        logger.info(f"Classification class names: {class_names}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def classify_bird(image):
    try:
        input_shape = cls_model.input_shape[1:3]  # Get height and width
        logger.info(f"Model expected input shape: {input_shape}")

        # Ensure image is in RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Input image shape before resize: {image.shape}")

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
        conf = float(np.max(output_data[0]))  # Convert to float for display

        return bird_species, conf
    except Exception as e:
        logger.error(f"Error in classify_bird: {str(e)}")
        return "Unknown", 0.0


def process_image(image):
    try:
        # Convert image to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detect birds using YOLO
        det_result = det_model.predict(rgb_image, imgsz=192, classes=14, conf=0.5)

        for result in det_result:
            for box in result.boxes:
                box = box.xyxy
                x1, y1, x2, y2 = map(int, box[0])
                logger.info(f"Detection box: ({x1}, {y1}, {x2}, {y2})")

                # Extract and classify cropped region
                crop = image[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    logger.warning("Invalid crop detected, skipping")
                    continue

                # Save crop for debugging
                cv2.imwrite("crop.png", crop)
                logger.info("Saved cropped image as crop.png")

                bird_species, conf = classify_bird(crop)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label and confidence
                label = f"{bird_species}: {conf:.2%}"
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return image


def main():
    # Get image path from command-line argument or set directly
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "assets/test-images/Pamarayeg_IIIx2_(cropped).jpg"  # Replace with your image path

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Error: Could not load image from {image_path}")
        return

    # Start model loading in a separate thread
    model_loading_thread = threading.Thread(target=load_model)
    model_loading_thread.start()
    model_loading_thread.join()  # Wait for models to load

    # Process image
    processed_image = process_image(image)

    # Save output
    output_path = "output.png"
    cv2.imwrite(output_path, processed_image)
    logger.info(f"Output image saved as {output_path}")

    # Display image
    window_name = "Endangered Bird Identifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, processed_image)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
