from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import requests as req_lib
from urllib.parse import quote

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL    = "llama-3.1-8b-instant"
THINK_MODEL   = "llama-3.3-70b-versatile"

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
    think     = data.get("think", False)
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
    """Generate an image via Pollinations.ai (free, no key needed).
    Uses Groq to rewrite the user's request into a clean image prompt."""
    data    = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])   # recent text history for context

    if not message:
        return jsonify({"error": "No message"}), 400

    # Build context string from recent history (text only, last 6 turns)
    context_lines = []
    for m in history[-6:]:
        if isinstance(m.get("content"), str):
            role = "User" if m["role"] == "user" else "Glitch"
            context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines)

    # Use Groq to extract a clean, detailed image generation prompt
    try:
        system = (
            "You are a prompt engineer for AI image generation. "
            "Given the conversation context and the user's latest request, "
            "write a single, vivid, descriptive image generation prompt (max 80 words). "
            "Output ONLY the prompt — no explanation, no quotes, no extra text."
        )
        user_msg = f"Conversation:\n{context}\n\nUser's request: {message}" if context else f"User's request: {message}"

        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            max_tokens=120
        )
        prompt = resp.choices[0].message.content.strip().strip('"').strip("'")
    except Exception as e:
        # Fallback: use message as-is
        prompt = message

    seed = int.from_bytes(os.urandom(4), "big")
    encoded = quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=768&nologo=true&seed={seed}"
    return jsonify({"url": url, "prompt": prompt})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
