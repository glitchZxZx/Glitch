from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import base64
import requests as req_lib

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL    = "llama-3.1-8b-instant"
THINK_MODEL   = "llama-3.3-70b-versatile"   # "think harder" — bigger, slower
IMAGE_MODEL   = "fal-ai/flux/schnell"        # fal.ai image gen (fixed path)

def load_personality():
    try:
        with open("personality.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        return "\n".join(lines)
    except Exception:
        return "You are a helpful assistant named Glitch."

PERSONALITY = load_personality()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/imprint")
def imprint():
    return render_template("imprint.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    history   = data.get("history", [])
    think     = data.get("think", False)      # think harder flag
    has_image = any(isinstance(m.get("content"), list) for m in history)

    if think:
        model = THINK_MODEL
    elif has_image:
        model = VISION_MODEL
    else:
        model = TEXT_MODEL

    messages = [{"role": "system", "content": PERSONALITY}] + history

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=2048 if think else 1024
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield f"(Error: {e})"

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/imagine", methods=["POST"])
def imagine():
    """Generate an image via fal.ai FLUX and return the image URL."""
    data   = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt"}), 400

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        return jsonify({"error": "FAL_KEY not set"}), 500

    try:
        # fal.ai REST API
        resp = req_lib.post(
            f"https://fal.run/{IMAGE_MODEL}",
            headers={"Authorization": f"Key {fal_key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "image_size": "landscape_4_3", "num_inference_steps": 4, "num_images": 1},
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        url = result["images"][0]["url"]
        return jsonify({"url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
