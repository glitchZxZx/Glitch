from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import re
import requests as req_lib
from urllib.parse import quote
from datetime import date
import time
from collections import deque

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Per-IP rate limit: max 20 requests per 60 seconds
RATE_LIMIT = 20
RATE_WINDOW = 60
_rate_store: dict[str, deque] = {}

def is_rate_limited(ip: str) -> bool:
    now = time.time()
    if ip not in _rate_store:
        _rate_store[ip] = deque()
    dq = _rate_store[ip]
    while dq and now - dq[0] > RATE_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return True
    dq.append(now)
    return False

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.1-8b-instant"
SCOUT_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
THINK_MODEL  = "llama-3.3-70b-versatile"

def load_personality():
    try:
        with open("personality.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        return "\n".join(lines)
    except Exception:
        return "You are a helpful assistant named Glitch."

PERSONALITY = load_personality()


def select_model(user_message, has_image, think):
    if has_image:
        return VISION_MODEL
    if len(user_message) < 50 and not think:
        return TEXT_MODEL
    if len(user_message) < 300 and not think:
        return SCOUT_MODEL
    return THINK_MODEL if think else SCOUT_MODEL


def get_max_tokens(model, think):
    if model == THINK_MODEL:
        return 768
    if model == SCOUT_MODEL:
        return 512
    return 384


# ── Local search pre-filter ──────────────────────────────────────────────────
# Avoids a Groq API call for the vast majority of casual messages.
# Only triggers the needs_search Groq call when there's a real search signal.

_CASUAL_STARTERS = {
    "hi", "hey", "hello", "sup", "yo", "hiya", "howdy",
    "thanks", "thank", "ty", "thx",
    "lol", "lmao", "haha", "hehe",
    "ok", "okay", "k", "sure", "yep", "yeah", "yup", "nope", "nah", "no", "yes",
    "cool", "nice", "great", "awesome", "wow", "omg",
    "good", "bad", "fine", "alright",
    "bye", "cya", "later",
}

_SEARCH_RE = re.compile(
    r"\b(who is|who are|what is|what are|when did|when is|when was|"
    r"where is|where are|how much|how many|price of|cost of|"
    r"latest|recent|current|right now|as of|today|this week|this year|"
    r"news|weather|score|standings|winner|champion|"
    r"release date|out now|launched|available now|"
    r"stock|crypto|bitcoin|exchange rate|"
    r"live|streaming|playing now)\b",
    re.IGNORECASE,
)

_CODE_RE = re.compile(r"(def |class |import |function |=>|===|!==|```)")


def needs_search(user_message: str):
    """
    Fast local pre-filter. Only calls Groq when there's an actual search signal.
    For casual chat (the majority of messages) this returns None with zero API calls.
    """
    msg   = user_message.strip()
    lower = msg.lower()
    words = lower.split()

    # Short messages never need search
    if len(words) <= 3:
        return None

    # Casual openers
    if words[0] in _CASUAL_STARTERS and len(words) <= 6:
        return None

    # Code/math content
    if _CODE_RE.search(msg):
        return None

    # No search keywords → skip API call entirely
    if not _SEARCH_RE.search(lower):
        return None

    # Has a search signal → ask the lightweight model for the actual query
    today = date.today().strftime("%B %d, %Y")
    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Today is {today}. "
                        "Does this need a live web search? "
                        "If yes reply: SEARCH: <concise query>  "
                        "If no reply: NO  "
                        "Nothing else."
                    )
                },
                {"role": "user", "content": msg}
            ],
            max_tokens=20
        )
        result = resp.choices[0].message.content.strip()
        if result.upper().startswith("SEARCH:"):
            return result[7:].strip()
        return None
    except Exception:
        return None


def web_search(query, max_results=3):
    try:
        api_key = os.environ.get("TAVILY_API_KEY")
        resp = req_lib.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic"
            },
            timeout=8
        )
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "body":  r.get("content", ""),
                "href":  r.get("url", "")
            }
            for r in data.get("results", [])
        ]
    except Exception:
        return []


def format_search_results(results):
    parts = []
    for r in results[:3]:
        title = r.get("title", "")
        body  = r.get("body", "")[:150]
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
    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    if is_rate_limited(ip):
        return Response(
            "You're sending messages too fast — give it a moment 🙂",
            mimetype="text/plain",
            status=429
        )

    data      = request.get_json()
    history   = data.get("history", [])
    think     = data.get("think", False)
    has_image = any(isinstance(m.get("content"), list) for m in history)

    # Trim history server-side to cap token usage
    if len(history) > 20:
        history = history[-20:]

    last_user_msg = ""
    for m in reversed(history):
        if m["role"] == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"]
            break

    model      = select_model(last_user_msg, has_image, think)
    max_tokens = get_max_tokens(model, think)

    def generate():
        try:
            search_context = ""
            search_query   = None

            if last_user_msg and not has_image:
                search_query = needs_search(last_user_msg)

            if search_query:
                yield f"§SEARCH:{search_query}§\n"
                results        = web_search(search_query)
                search_context = format_search_results(results) if results else ""

            system    = PERSONALITY
            today_str = date.today().strftime("%B %d, %Y")

            if search_context:
                system += (
                    f"\n\n[Web search results for '{search_query}']\n"
                    f"{search_context}\n"
                    "[End of search results]\n\n"
                    f"Today is {today_str}. "
                    "You have real web search results above. "
                    "Answer using these — do NOT say you lack real-time info. "
                    "Be direct. Cite the URL when helpful."
                )
            elif search_query:
                system += (
                    f"\n\nToday is {today_str}. "
                    f"A web search for '{search_query}' returned nothing. "
                    "Answer from training knowledge and be honest about it."
                )

            messages = [{"role": "system", "content": system}] + history

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                yield "(Groq is busy right now — try again in a moment)"
            else:
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


def print_banner():
    print("""
\033[35m
  ██████  ██      ██ ████████  ██████ ██   ██
 ██       ██      ██    ██    ██      ██   ██
 ██   ███ ██      ██    ██    ██      ███████
 ██    ██ ██      ██    ██    ██      ██   ██
  ██████  ███████ ██    ██     ██████ ██   ██
\033[0m
  \033[90mv1.3 · single-call · local search filter · rate-limited\033[0m
""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print_banner()
    app.run(host="0.0.0.0", port=port, debug=False)
