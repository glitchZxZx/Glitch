from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import re
import json
import requests as req_lib
from urllib.parse import quote
from datetime import date
import time
from collections import deque
import smtplib
import random
from email.mime.text import MIMEText

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Simple IP → username store (in-memory, survives restarts via JSON file)
USERNAME_FILE = "usernames.json"

def _load_usernames():
    try:
        with open(USERNAME_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_usernames(d):
    try:
        with open(USERNAME_FILE, "w") as f:
            json.dump(d, f)
    except Exception:
        pass

_usernames: dict = _load_usernames()

# ── Server-side chat storage ─────────────────────────────────────────────────
CHATS_FILE = "chats.json"

def _load_chats():
    try:
        with open(CHATS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_chats(d):
    try:
        with open(CHATS_FILE, "w") as f:
            json.dump(d, f)
    except Exception:
        pass

_chats: dict = _load_chats()

@app.route("/api/chats", methods=["GET"])
def get_chats():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    # Only return chats if user has a username (is "logged in")
    if not _usernames.get(ip):
        return jsonify({"chats": [], "loggedIn": False})
    return jsonify({"chats": _chats.get(ip, []), "loggedIn": True})

@app.route("/api/chats", methods=["POST"])
def save_chat():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    if not _usernames.get(ip):
        return jsonify({"error": "not logged in"}), 401
    data = request.get_json()
    chat = data.get("chat")
    if not chat or not chat.get("id"):
        return jsonify({"error": "invalid"}), 400
    user_chats = _chats.get(ip, [])
    # Update existing or insert at front
    for i, c in enumerate(user_chats):
        if c["id"] == chat["id"]:
            user_chats[i] = chat
            break
    else:
        user_chats.insert(0, chat)
    _chats[ip] = user_chats[:50]  # max 50 chats per user
    _save_chats(_chats)
    return jsonify({"ok": True})

@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat_route(chat_id):
    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    if not _usernames.get(ip):
        return jsonify({"error": "not logged in"}), 401
    user_chats = _chats.get(ip, [])
    _chats[ip] = [c for c in user_chats if c["id"] != chat_id]
    _save_chats(_chats)
    return jsonify({"ok": True})


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

SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def load_personality():
    try:
        with open("personality.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        return "\n".join(lines)
    except Exception:
        return "You are a helpful assistant named Glitch."

PERSONALITY = load_personality()

_CASUAL_STARTERS = {
    "hi", "hey", "hello", "sup", "yo", "hiya", "howdy",
    "thanks", "thank", "ty", "thx",
    "lol", "lmao", "haha", "hehe",
    "ok", "okay", "k", "sure", "yep", "yeah", "yup", "nope", "nah", "no", "yes",
    "cool", "nice", "great", "awesome", "wow", "omg",
    "good", "bad", "fine", "alright",
    "bye", "cya", "later",
}

_CODE_RE = re.compile(r"(def |class |import |function |=>|===|!==|```)")

_OPINION_RE = re.compile(
    r"^(what'?s? your|what do you (think|feel|believe)|do you (think|believe|feel|like|prefer)|"
    r"ur opinion|your opinion|how do you feel|would you rather|fav(ou?rite)?|"
    r"do you ever|have you ever|are you|were you|will you|can you imagine|"
    r"what would you|if you (were|could|had))",
    re.IGNORECASE,
)

_SEARCH_SIGNALS = re.compile(
    r"\b(who is|who are|what is|what are|when did|when is|when was|"
    r"where is|where are|how much|how many|price of|cost of|"
    r"latest|recent|current|right now|as of|today|this week|this year|"
    r"news|weather|score|standings|winner|champion|"
    r"release date|out now|launched|available|"
    r"stock|crypto|bitcoin|exchange rate|"
    r"best .{0,30}right now|top .{0,30}(now|today|2024|2025)|"
    r"recommend|should i watch|what('?s| is) good|worth (watching|reading|playing))\b",
    re.IGNORECASE,
)


def needs_search(user_message: str):
    msg   = user_message.strip()
    lower = msg.lower()
    words = lower.split()

    if len(words) <= 2:
        return None
    if words[0] in _CASUAL_STARTERS and len(words) <= 5:
        return None
    if all(w in _CASUAL_STARTERS for w in words):
        return None
    if _CODE_RE.search(msg):
        return None
    if _OPINION_RE.search(lower):
        return None
    if len(words) <= 8 and not any(c in lower for c in ["?", "who", "what", "when", "where", "how", "which"]):
        return None
    if _SEARCH_SIGNALS.search(lower):
        return msg[:120]
    if len(words) >= 5 and "?" in msg:
        return msg[:120]

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


# ── OTP email system ──────────────────────────────────────────────────────────
_codes: dict = {}

GMAIL_ADDRESS  = os.environ.get("GMAIL_ADDRESS", "glitch.l.l.m.ai@gmail.com")
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

def send_otp_email(to_email: str, code: str):
    msg = MIMEText(
        f"Your Glitch verification code is:\n\n"
        f"  {code}\n\n"
        f"This code expires in 10 minutes.\n\n"
        f"If you didn't request this, you can ignore this email."
    )
    msg["Subject"] = f"{code} is your Glitch code"
    msg["From"]    = f"Glitch <{GMAIL_ADDRESS}>"
    msg["To"]      = to_email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_ADDRESS, GMAIL_PASSWORD)
        s.send_message(msg)

@app.route("/api/send-code", methods=["POST"])
def api_send_code():
    data  = request.get_json()
    email = (data or {}).get("email", "").strip().lower()

    if not email or "@" not in email:
        return jsonify({"error": "invalid email"}), 400

    code = str(random.randint(100000, 999999))
    _codes[email] = (code, time.time() + 600)

    try:
        send_otp_email(email, code)
    except Exception as e:
        print(f"[OTP] Failed to send to {email}: {e}")
        return jsonify({"error": "Failed to send email. Check server config."}), 500

    return jsonify({"ok": True})

@app.route("/api/verify-code", methods=["POST"])
def api_verify_code():
    ip      = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    data    = request.get_json()
    email   = (data or {}).get("email", "").strip().lower()
    entered = (data or {}).get("code", "").strip()

    record = _codes.get(email)
    if not record:
        return jsonify({"ok": False, "error": "No code sent for this email"}), 400
    stored_code, expires_at = record
    if time.time() > expires_at:
        del _codes[email]
        return jsonify({"ok": False, "error": "Code expired"}), 400
    if entered != stored_code:
        return jsonify({"ok": False, "error": "Wrong code"}), 400

    del _codes[email]
    # Tell the frontend whether this user already has a username set
    existing_username = _usernames.get(ip, "")
    return jsonify({"ok": True, "hasUsername": bool(existing_username), "username": existing_username})


# ── Username routes ───────────────────────────────────────────────────────────
@app.route("/api/username", methods=["GET"])
def get_username():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    return jsonify({"username": _usernames.get(ip, "")})

@app.route("/api/username", methods=["POST"])
def set_username():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    data = request.get_json()
    name = (data.get("username") or "").strip()[:32]
    if not name:
        return jsonify({"error": "empty"}), 400
    import re as _re
    name = _re.sub(r"[^\w\s\-\.!]", "", name)[:32].strip()
    if not name:
        return jsonify({"error": "invalid"}), 400
    _usernames[ip] = name
    _save_usernames(_usernames)
    return jsonify({"username": name})


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/signup")
def signup_page():
    # Renders the same login template but with ?new=1 flag baked in
    return render_template("login.html", signup=True)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/imprint")
def imprint():
    return render_template("imprint.html")


# ── AI title generation ───────────────────────────────────────────────────────
@app.route("/api/title", methods=["POST"])
def generate_title():
    data      = request.get_json()
    user_msg  = (data.get("userMsg") or "")[:200]
    ai_msg    = (data.get("aiMsg")   or "")[:200]
    if not user_msg:
        return jsonify({"title": "New chat"})
    try:
        resp = client.chat.completions.create(
            model=SCOUT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Generate a very short chat title (3–5 words max). "
                    "No quotes, no punctuation at end, no emoji. "
                    "Output ONLY the title — nothing else."
                )},
                {"role": "user", "content": f"User: {user_msg}\nAssistant: {ai_msg}"}
            ],
            max_tokens=16
        )
        title = resp.choices[0].message.content.strip().strip("\"'").strip()
        if len(title) > 52:
            title = title[:52]
        return jsonify({"title": title or user_msg[:32]})
    except Exception:
        short = user_msg[:32] + ("…" if len(user_msg) > 32 else "")
        return jsonify({"title": short})


# ── Chat route ────────────────────────────────────────────────────────────────
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

    if len(history) > 20:
        history = history[-20:]

    last_user_msg = ""
    for m in reversed(history):
        if m["role"] == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"]
            break

    max_tokens = 768 if think else 512

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

            today_str = date.today().strftime("%B %d, %Y")
            system    = PERSONALITY + f"\n\nToday is {today_str}."

            if _CODE_RE.search(last_user_msg) or any(w in last_user_msg.lower() for w in ["code", "fix", "debug", "function", "script", "error", "bug", "write a"]):
                system += "\n\nThe user is asking about code. Be precise, use code blocks with the correct language tag, and keep explanations short. Working code over lengthy explanation."

            if search_context:
                system += (
                    f"\n\n[Web search results for '{search_query}']\n"
                    f"{search_context}\n"
                    "[End of search results]\n\n"
                    "You have real web search results above. "
                    "Answer using these — do NOT say you lack real-time info. "
                    "Talk like a person, not like you're reading search results. "
                    "Don't cite sources or mention URLs unless the user specifically asks for them."
                )
            elif search_query:
                system += (
                    f"\n\nA web search for '{search_query}' returned nothing. "
                    "Answer from training knowledge and be honest about it."
                )

            messages = [{"role": "system", "content": system}] + history

            stream = client.chat.completions.create(
                model=SCOUT_MODEL,
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
            model=SCOUT_MODEL,
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
  \033[90mv1.8 · server-side chats · Gmail SMTP\033[0m
""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print_banner()
    app.run(host="0.0.0.0", port=port, debug=False)