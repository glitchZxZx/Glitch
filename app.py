from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import json
import requests as req_lib
from urllib.parse import quote

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.1-8b-instant"
THINK_MODEL  = "llama-3.3-70b-versatile"

def load_personality():
    try:
        with open("personality.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        return "\n".join(lines)
    except Exception:
        return "You are a helpful assistant named Glitch."

PERSONALITY = load_personality()

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this for recent news, events, "
            "prices, weather, sports scores, or anything requiring up-to-date data. "
            "Do NOT use for general knowledge or things you already know well."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A concise search query"}
            },
            "required": ["query"]
        }
    }
}

def web_search(query, max_results=5):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        return [{"title": "Search unavailable", "body": str(e), "href": ""}]

def format_search_results(results):
    parts = []
    for r in results[:5]:
        title = r.get("title", "")
        body  = r.get("body", "")
        href  = r.get("href", "")
        parts.append(f"Title: {title}\nSummary: {body}\nURL: {href}")
    return "\n\n---\n\n".join(parts)


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
    data      = request.get_json()
    history   = data.get("history", [])
    think     = data.get("think", True)
    has_image = any(isinstance(m.get("content"), list) for m in history)

    if has_image:
        model = VISION_MODEL
    else:
        model = THINK_MODEL if think else TEXT_MODEL

    messages = [{"role": "system", "content": PERSONALITY}] + history

    def generate():
        try:
            # First pass — check if model wants to search
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[SEARCH_TOOL],
                tool_choice="auto",
                max_tokens=2048 if think else 1024
            )

            msg = resp.choices[0].message

            if msg.tool_calls:
                tool_messages = []
                for tc in msg.tool_calls:
                    if tc.function.name == "web_search":
                        args  = json.loads(tc.function.arguments)
                        query = args.get("query", "")

                        # Signal the frontend to show search pill
                        yield f"§SEARCH:{query}§\n"

                        results      = web_search(query)
                        results_text = format_search_results(results)

                        tool_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": results_text
                        })

                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                }

                messages2 = messages + [assistant_msg] + tool_messages

                stream = client.chat.completions.create(
                    model=model,
                    messages=messages2,
                    stream=True,
                    max_tokens=2048
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            elif msg.content:
                yield msg.content

            else:
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
    data    = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "No message"}), 400

    context_lines = []
    for m in history[-6:]:
        if isinstance(m.get("content"), str):
            role = "User" if m["role"] == "user" else "Glitch"
            context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines)

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
        prompt = message

    seed    = int.from_bytes(os.urandom(4), "big")
    encoded = quote(prompt)
    url     = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=768&nologo=true&seed={seed}"
    return jsonify({"url": url, "prompt": prompt})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
