from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import json
import requests as req_lib
from urllib.parse import quote
from datetime import date

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


def needs_search(user_message):
    """Ask the fast model if this needs a web search. Returns query string or None."""
    today = date.today().strftime("%B %d, %Y")
    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Today is {today}. "
                        "You decide if a question would benefit from a web search. "
                        "Default to YES unless it is clearly a personal, creative, or coding task, or pure opinion. "
                        "Search for: news, events, scores, weather, prices, population, statistics, "
                        "facts about places/people/companies, anything that could be outdated, "
                        "or any factual question where fresher data helps. "
                        "When writing the search query, use the current year where relevant. "
                        "If yes, reply exactly: SEARCH: <concise query> "
                        "If no, reply exactly: NO "
                        "Nothing else."
                    )
                },
                {"role": "user", "content": user_message}
            ],
            max_tokens=25
        )
        result = resp.choices[0].message.content.strip()
        if result.upper().startswith("SEARCH:"):
            return result[7:].strip()
        return None
    except Exception:
        return None


SEARX_INSTANCES = [
    "https://searx.be",
    "https://search.bus-hit.me",
    "https://searxng.world",
]

def web_search(query, max_results=5):
    # Try duckduckgo_search first
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            return results
    except Exception:
        pass

    # Fallback: try public SearXNG instances
    for base in SEARX_INSTANCES:
        try:
            resp = req_lib.get(
                f"{base}/search",
                params={"q": query, "format": "json", "categories": "general"},
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Glitch-bot/1.0)"}
            )
            data = resp.json()
            results = [
                {
                    "title": r.get("title", ""),
                    "body":  r.get("content", ""),
                    "href":  r.get("url", "")
                }
                for r in data.get("results", [])[:max_results]
                if r.get("content")
            ]
            if results:
                return results
        except Exception:
            continue

    return []  # caller handles empty


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

    # Get last user message (text only) for search decision
    last_user_msg = ""
    for m in reversed(history):
        if m["role"] == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"]
            break

    def generate():
        try:
            search_context = ""
            search_query   = None

            # Only check for search on text messages
            if last_user_msg and not has_image:
                search_query = needs_search(last_user_msg)

            if search_query:
                yield f"§SEARCH:{search_query}§\n"
                results        = web_search(search_query)
                search_context = format_search_results(results) if results else ""

            # Build system prompt — inject search results if available
            system = PERSONALITY
            today_str = date.today().strftime("%B %d, %Y")
            if search_context:
                system += (
                    f"\n\n[Web search results for '{search_query}']\n"
                    f"{search_context}\n"
                    "[End of search results]\n\n"
                    f"Today is {today_str}. "
                    "IMPORTANT: You have real web search results above. "
                    "You MUST answer using these results — do NOT say you lack real-time information. "
                    "Be direct and factual. Cite the source URL when helpful."
                )
            elif search_query:
                # Search ran but returned nothing
                system += (
                    f"\n\nToday is {today_str}. "
                    f"A web search for '{search_query}' returned no results. "
                    "Answer from your training knowledge and be upfront that you couldn't fetch live data."
                )

            messages = [{"role": "system", "content": system}] + history

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
