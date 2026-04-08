from flask import Flask, render_template, request, Response, stream_with_context
from groq import Groq
import os
import base64

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.1-8b-instant"

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
    history = data.get("history", [])
    has_image = any(
        isinstance(m.get("content"), list)
        for m in history
    )
    model = VISION_MODEL if has_image else TEXT_MODEL

    messages = [{"role": "system", "content": PERSONALITY}] + history

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=1024
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            yield f"(Error: {e})"

    return Response(stream_with_context(generate()), mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
