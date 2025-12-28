import io
import numpy as np
from PIL import Image
import tensorflow as tf


def load_image_from_bytes(image_bytes: bytes, target_size=(512, 512)) -> tf.Tensor:
    """
    Load image from bytes, resize, normalize [0,1], add batch dimension.
    Returns tensor of shape (1, H, W, 3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)

    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
    return tensor


def tensor_to_pil_image(tensor: tf.Tensor) -> Image.Image:
    """
    Convert (1, H, W, 3) or (H, W, 3) tensor in [0,1] to PIL Image.
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    arr = (tensor.numpy() * 255.0).astype("uint8")
    img = Image.fromarray(arr)
    return img
