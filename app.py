from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from preprocessor import load_image_from_bytes, tensor_to_pil_image
from utils import load_nst_model, run_style_transfer, pil_to_base64_str

app = Flask(__name__, template_folder="templates")

print("Loading Neural Style Transfer model...")
model = load_nst_model("model")
print("Model loaded.")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api/style-transfer", methods=["POST"])
def style_transfer():
    if "content_image" not in request.files or "style_image" not in request.files:
        resp = jsonify({"error": "Both 'content_image' and 'style_image' files are required."})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 400

    content_file = request.files["content_image"]
    style_file = request.files["style_image"]

    try:
        content_bytes = content_file.read()
        style_bytes = style_file.read()

        content_tensor = load_image_from_bytes(content_bytes, target_size=(512, 512))
        style_tensor = load_image_from_bytes(style_bytes, target_size=(512, 512))

        output_tensor = run_style_transfer(model, content_tensor, style_tensor)

        output_image = tensor_to_pil_image(output_tensor)
        image_b64 = pil_to_base64_str(output_image, format="PNG")

        resp = jsonify({"image_base64": image_b64})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 200

    except Exception as e:
        print("Error during style transfer:", e)
        resp = jsonify({"error": "Failed to run style transfer.", "details": str(e)})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 500


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
    return response


if __name__ == "__main__":
    # pip install flask tensorflow pillow
    app.run(host="0.0.0.0", port=5000, debug=True)
