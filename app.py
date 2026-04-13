from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from groq import Groq
import os
import re
import json
import secrets
import requests as req_lib
from urllib.parse import quote
from datetime import date
import time
from collections import deque
import random

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Supabase client ───────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")        # e.g. https://xyz.supabase.co
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")   # anon/public key

def _sb_headers(prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h

def sb_get(table, params=None):
    try:
        r = req_lib.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers(),
            params=params,
            timeout=5
        )
        return r.json() if r.ok else []
    except Exception:
        return []

def sb_upsert(table, data):
    try:
        r = req_lib.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers("resolution=merge-duplicates,return=representation"),
            json=data,
            timeout=5
        )
        return r.ok
    except Exception:
        return False

def sb_insert(table, data):
    try:
        r = req_lib.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers("return=representation"),
            json=data,
            timeout=5
        )
        return r.ok
    except Exception:
        return False

def sb_delete(table, params):
    try:
        r = req_lib.delete(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers(),
            params=params,
            timeout=5
        )
        return r.ok
    except Exception:
        return False

# ── Session auth ──────────────────────────────────────────────────────────────
def get_session_email(req):
    """Validate X-Session-Token header, return email or None."""
    token = req.headers.get("X-Session-Token", "").strip()
    if not token or len(token) > 100:
        return None
    rows = sb_get("sessions", {"token": f"eq.{token}", "select": "email"})
    if rows and isinstance(rows, list) and len(rows) > 0:
        return rows[0].get("email")
    return None

# ── OTP storage & rate limiting ───────────────────────────────────────────────
_codes: dict = {}          # email -> (code, expires_at)
_otp_attempts: dict = {}   # email -> [timestamps]
OTP_MAX_ATTEMPTS = 5
OTP_WINDOW = 600  # 10 minutes

def check_otp_attempts(email: str) -> bool:
    """Returns True if the email is temporarily blocked."""
    now = time.time()
    attempts = [t for t in _otp_attempts.get(email, []) if now - t < OTP_WINDOW]
    _otp_attempts[email] = attempts
    return len(attempts) >= OTP_MAX_ATTEMPTS

def record_otp_attempt(email: str):
    now = time.time()
    attempts = [t for t in _otp_attempts.get(email, []) if now - t < OTP_WINDOW]
    attempts.append(now)
    _otp_attempts[email] = attempts

def clear_otp_attempts(email: str):
    _otp_attempts.pop(email, None)

# ── Chat rate limiting ────────────────────────────────────────────────────────
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

# ── History validation ────────────────────────────────────────────────────────
def validate_history(history):
    """Strip system-role injections, cap length, limit content size."""
    if not isinstance(history, list):
        return []
    clean = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue  # silently drop any system-role injection attempts
        content = m.get("content")
        if isinstance(content, str):
            content = content[:8000]  # cap individual message length
        elif isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts).strip() or "[image]"
        else:
            continue
        clean.append({"role": role, "content": content})
    return clean[-20:]  # max 20 messages

# ── Model & personality ───────────────────────────────────────────────────────
SCOUT_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "llama-3.3-70b-versatile"

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


# ── OTP email (Brevo) ─────────────────────────────────────────────────────────
BREVO_API_KEY    = os.environ.get("BREVO_API_KEY", "")
SENDER_EMAIL     = os.environ.get("SENDER_EMAIL", "glitch.l.l.m.ai@gmail.com")
SENDER_NAME      = os.environ.get("SENDER_NAME", "Glitch")

def send_otp_email(to_email: str, code: str):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"/></head>
<body style="margin:0;padding:0;background:#0e0e11;font-family:-apple-system,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0e0e11;padding:40px 16px;">
    <tr><td align="center">
      <table width="100%" cellpadding="0" cellspacing="0" style="max-width:480px;">
        <tr><td style="padding-bottom:28px;text-align:center;">
          <span style="font-size:28px;font-weight:800;color:#e8e8f0;letter-spacing:-0.03em;">Glitch</span>
        </td></tr>
        <tr><td style="background:#16161a;border:1px solid #2a2a35;border-radius:16px;padding:36px 32px;">
          <p style="margin:0 0 8px;font-size:15px;font-weight:600;color:#e8e8f0;">Your verification code</p>
          <p style="margin:0 0 28px;font-size:13px;color:#6b6b80;line-height:1.6;">
            Enter this code on the sign-up page. It expires in 10 minutes.
          </p>
          <div style="background:#0e0e11;border:1px solid #2a2a35;border-radius:12px;padding:24px;text-align:center;margin-bottom:28px;">
            <span style="font-size:38px;font-weight:800;letter-spacing:0.15em;color:#7c6ef0;">{code}</span>
          </div>
          <p style="margin:0;font-size:12px;color:#6b6b80;text-align:center;">
            If you didn't request this, you can safely ignore this email.
          </p>
        </td></tr>
        <tr><td style="padding-top:20px;text-align:center;">
          <span style="font-size:12px;color:#3a3a50;">Glitch &middot; <a href="https://glitch-ozuf.onrender.com" style="color:#7c6ef0;text-decoration:none;">glitch-ozuf.onrender.com</a></span>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    resp = req_lib.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
            "to": [{"email": to_email}],
            "subject": f"{code} is your Glitch code",
            "htmlContent": html
        },
        timeout=10
    )
    if not resp.ok:
        raise Exception(f"Brevo error {resp.status_code}: {resp.text}")


# ── OTP routes ────────────────────────────────────────────────────────────────
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
    data    = request.get_json()
    email   = (data or {}).get("email", "").strip().lower()
    entered = (data or {}).get("code", "").strip()

    # OTP brute-force protection
    if check_otp_attempts(email):
        return jsonify({"ok": False, "error": "Too many attempts. Try again in 10 minutes."}), 429

    record_otp_attempt(email)

    record = _codes.get(email)
    if not record:
        return jsonify({"ok": False, "error": "No code sent for this email"}), 400
    stored_code, expires_at = record
    if time.time() > expires_at:
        del _codes[email]
        return jsonify({"ok": False, "error": "Code expired"}), 400
    if entered != stored_code:
        return jsonify({"ok": False, "error": "Wrong code"}), 400

    # Success — clear code and attempt counter
    del _codes[email]
    clear_otp_attempts(email)

    # Ensure user row exists
    sb_upsert("users", {"email": email})

    # Get existing username
    rows = sb_get("users", {"email": f"eq.{email}", "select": "username"})
    existing_username = (rows[0].get("username") or "") if rows else ""

    # Generate a secure session token
    token = secrets.token_urlsafe(32)
    sb_insert("sessions", {"token": token, "email": email})

    return jsonify({
        "ok": True,
        "token": token,
        "hasUsername": bool(existing_username),
        "username": existing_username
    })


# ── Username routes ───────────────────────────────────────────────────────────
@app.route("/api/username", methods=["GET"])
def get_username():
    email = get_session_email(request)
    if not email:
        return jsonify({"username": ""})
    rows = sb_get("users", {"email": f"eq.{email}", "select": "username"})
    username = (rows[0].get("username") or "") if rows else ""
    return jsonify({"username": username})


@app.route("/api/username", methods=["POST"])
def set_username():
    email = get_session_email(request)
    if not email:
        return jsonify({"error": "not logged in"}), 401
    data = request.get_json()
    name = (data.get("username") or "").strip()[:32]
    if not name:
        return jsonify({"error": "empty"}), 400
    name = re.sub(r"[^\w\s\-\.!]", "", name)[:32].strip()
    if not name:
        return jsonify({"error": "invalid"}), 400
    sb_upsert("users", {"email": email, "username": name})
    return jsonify({"username": name})


# ── Chat storage routes ───────────────────────────────────────────────────────
@app.route("/api/chats", methods=["GET"])
def get_chats():
    email = get_session_email(request)
    if not email:
        return jsonify({"chats": [], "loggedIn": False})
    rows = sb_get("chats", {
        "email": f"eq.{email}",
        "select": "data",
        "order": "updated_at.desc",
        "limit": "50"
    })
    chats = [r["data"] for r in rows if "data" in r]
    return jsonify({"chats": chats, "loggedIn": True})


@app.route("/api/chats", methods=["POST"])
def save_chat():
    email = get_session_email(request)
    if not email:
        return jsonify({"error": "not logged in"}), 401
    data = request.get_json()
    chat = data.get("chat")
    if not chat or not chat.get("id"):
        return jsonify({"error": "invalid"}), 400
    ok = sb_upsert("chats", {
        "id": chat["id"],
        "email": email,
        "data": chat,
        "updated_at": date.today().isoformat()
    })
    return jsonify({"ok": ok})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat_route(chat_id):
    email = get_session_email(request)
    if not email:
        return jsonify({"error": "not logged in"}), 401
    ok = sb_delete("chats", {"id": f"eq.{chat_id}", "email": f"eq.{email}"})
    return jsonify({"ok": ok})


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/signup")
def signup_page():
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
    raw_history = data.get("history", [])
    think     = data.get("think", False)

    # Validate and sanitize history
    history   = validate_history(raw_history)
    has_image = any(isinstance(m.get("content"), list) for m in history)

    last_user_msg = ""
    for m in reversed(history):
        if m["role"] == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"]
            break

    max_tokens = 1200 if think else 800

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

            def try_stream(model):
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens
                )

            try:
                stream = try_stream(SCOUT_MODEL)
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    print(f"[Fallback] {SCOUT_MODEL} rate limited, switching to {FALLBACK_MODEL}")
                    stream = try_stream(FALLBACK_MODEL)
                else:
                    raise

            # Stream chunks, stripping <think>...</think> blocks (Qwen3 reasoning tokens)
            in_think = False
            buf = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                buf += content
                # Process buffer to strip <think> blocks
                while True:
                    if in_think:
                        end = buf.find("</think>")
                        if end != -1:
                            buf = buf[end + len("</think>"):]
                            in_think = False
                        else:
                            buf = ""  # discard everything inside think block
                            break
                    else:
                        start = buf.find("<think>")
                        if start != -1:
                            if start > 0:
                                yield buf[:start]
                            buf = buf[start + len("<think>"):]
                            in_think = True
                        else:
                            # No think tag — safe to yield, but keep last 7 chars
                            # buffered in case a tag is split across chunks
                            if len(buf) > 7:
                                yield buf[:-7]
                                buf = buf[-7:]
                            break
            if buf and not in_think:
                yield buf

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                yield "(Groq is really busy right now — try again in a moment)"
            else:
                yield f"(Error: {e})"

    return Response(stream_with_context(generate()), mimetype="text/plain")


# ── Image generation route ────────────────────────────────────────────────────
@app.route("/imagine", methods=["POST"])
def imagine():
    data    = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "No message"}), 400

    context_lines = []
    for m in validate_history(history)[-6:]:
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
    except Exception:
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
  \033[90mv2.0 · Supabase · token auth · OTP rate limit\033[0m
""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print_banner()
    app.run(host="0.0.0.0", port=port, debug=False)
