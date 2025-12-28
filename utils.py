import os
import base64
import io
import tensorflow as tf
from PIL import Image


def load_nst_model(model_dir: str):
    """
    Load the TensorFlow SavedModel from a directory that contains:
      - saved_model.pb
      - variables/
    """
    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading SavedModel from: {model_dir}")
    model = tf.saved_model.load(model_dir)
    print("SavedModel loaded successfully.")
    return model


def run_style_transfer(model, content_tensor, style_tensor):
    """
    Run neural style transfer using the SavedModel.

    Assumes the model has a 'serving_default' signature taking
    content + style images and returning the stylized image.
    """
    infer = None
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        infer = model.signatures["serving_default"]

    if infer is not None:
        _, input_dict = infer.structured_input_signature
        input_names = list(input_dict.keys())

        feed = {}
        if len(input_names) >= 2:
            feed[input_names[0]] = content_tensor
            feed[input_names[1]] = style_tensor
        else:
            feed = {
                "content_image": content_tensor,
                "style_image": style_tensor,
            }

        outputs = infer(**feed)
    else:
        # fallback: directly call model
        outputs = model(content_tensor, style_tensor)

    if isinstance(outputs, dict):
        first_key = list(outputs.keys())[0]
        output_tensor = outputs[first_key]
    else:
        output_tensor = outputs

    return output_tensor


def pil_to_base64_str(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL image to base64 string (without data: prefix).
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64
