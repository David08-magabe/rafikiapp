"""
🇹🇿 Rafiki-Finance AI — Final Version
========================================
Stack:
  - Streamlit (UI)
  - Groq API + Llama 3.3 70B (LLM)
  - CSV knowledge base (150 Q&A pairs)
  - Bilingual: English + Kiswahili
  - Chat history (sidebar)
  - Text-to-Speech
  - Out-of-scope & gibberish detection
  - Onboarding: lang → tier → category → chat

Run:
    pip install streamlit pandas groq
    set GROQ_API_KEY=gsk_xxxxxxxxxxxx
    streamlit run rafiki_final.py
"""

import os
import json
from datetime import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Rafiki-Finance AI",
    page_icon="🇹🇿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
CSV_PATH      = "tz_financial_faq_150_matrix.csv"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"
BOT_AVATAR    = "🇹🇿"
BOT_AVATAR_URL = ""  

TIERS = {
    "English": [
        ("Informal/Machinga", "🛒", "Informal / Machinga"),
        ("Micro",             "🏪", "Micro business"),
        ("SME",               "🏢", "SME"),
        ("Formal Employee",   "👔", "Formal Employee"),
        ("Corporate",         "🏦", "Corporate"),
    ],
    "Kiswahili": [
        ("Informal/Machinga", "🛒", "Biashara Ndogo / Machinga"),
        ("Micro",             "🏪", "Biashara ya Micro"),
        ("SME",               "🏢", "SME"),
        ("Formal Employee",   "👔", "Mfanyakazi Rasmi"),
        ("Corporate",         "🏦", "Kampuni Kubwa"),
    ],
}

CATS = {
    "English": [
        ("Savings & Investments",              "💰", "Savings & Investments"),
        ("Loans & Credit",                     "🤝", "Loans & Credit"),
        ("Taxes & Regulation",                 "📋", "Taxes & Regulation"),
        ("Mobile Money & Digital Banking",     "📱", "Mobile Money & Banking"),
        ("Business Formalization & Insurance", "🛡️",  "Business & Insurance"),
    ],
    "Kiswahili": [
        ("Savings & Investments",              "💰", "Akiba na Uwekezaji"),
        ("Loans & Credit",                     "🤝", "Mikopo na Mkopo"),
        ("Taxes & Regulation",                 "📋", "Kodi na Kanuni"),
        ("Mobile Money & Digital Banking",     "📱", "Pesa ya Simu na Benki"),
        ("Business Formalization & Insurance", "🛡️",  "Usajili na Bima"),
    ],
}

DEPTH_MAP = {
    "Informal/Machinga": "Use very simple practical language. Reference M-Pesa (*150*00#), Tigo Pesa (*150*01#), Airtel Timiza (*150*60#), street-vendor realities.",
    "Micro":             "Use clear language. Reference FINCA, BRAC, CRDB Microfinance, NMB Kikundi, Lipa Namba merchant codes.",
    "SME":               "Use business-oriented language. Reference TRA EFD, BRELA, NMB, CRDB, NBC, VAT threshold TZS 100M.",
    "Formal Employee":   "Use professional language. Address PAYE bands, NSSF/NHIF, HESLB 15%, salary loans, 1/3 take-home rule.",
    "Corporate":         "Use corporate-level language. Address BOT, DSE listing, Transfer Pricing, Thin Capitalization, WHT, ESG.",
}

THANKS = {
    "Informal/Machinga": {"English": "Welcome, Machinga friend! Great to have you.",         "Kiswahili": "Karibu sana, ndugu Machinga!"},
    "Micro":             {"English": "Excellent! Micro businesses drive Tanzania's economy.", "Kiswahili": "Vizuri! Biashara ndogo ndizo nguvu ya Tanzania."},
    "SME":               {"English": "SME — the true backbone of our economy!",              "Kiswahili": "SME — nguvu halisi ya uchumi wetu!"},
    "Formal Employee":   {"English": "Welcome, valued formal employee!",                      "Kiswahili": "Karibu, mfanyakazi hodari!"},
    "Corporate":         {"English": "Welcome to the Corporate Finance level!",               "Kiswahili": "Karibu kwenye kiwango cha Corporate!"},
}

# CSS
st.markdown("""
<style>
.stApp { background: #e5ddd5 !important; }
.main .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
[data-testid="stSidebar"] {
    background: #075e54 !important;
    min-width: 280px !important;
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
.header-card {
    background: #075e54;
    border-radius: 14px 14px 0 0;
    padding: 12px 16px;
    display: flex; align-items: center; gap: 12px;
}
.header-avatar {
    width: 44px; height: 44px; border-radius: 50%;
    background: #128c7e;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.header-name { color: #fff; font-size: 15px; font-weight: 600; margin: 0; }
.header-sub  { color: #d0f0ec; font-size: 12px; margin: 2px 0 0; }
.online-dot  {
    width: 9px; height: 9px; border-radius: 50%;
    background: #25d366; display: inline-block; margin-right: 5px;
}
.chat-area {
    background: #e5ddd5;
    border-radius: 0 0 14px 14px;
    padding: 14px 16px;
    min-height: 0;
}
.bot-bubble {
    background: #ffffff;
    border-radius: 10px; border-top-left-radius: 0;
    padding: 9px 13px;
    margin: 4px 0;
    font-size: 13.5px; line-height: 1.55;
    box-shadow: 0 1px 2px rgba(0,0,0,.10);
    max-width: 82%; color: #111;
    display: inline-block;
}
.user-bubble {
    background: #dcf8c6;
    border-radius: 10px; border-top-right-radius: 0;
    padding: 9px 13px;
    margin: 4px 0 4px auto;
    font-size: 13.5px; line-height: 1.55;
    box-shadow: 0 1px 2px rgba(0,0,0,.10);
    max-width: 75%;
    min-width: 80px;
    width: fit-content;
    color: #111;
    display: block;
    text-align: left;
    word-wrap: break-word;
    word-break: break-word;
    white-space: pre-wrap;
    overflow-wrap: break-word;
}
.bubble-row-bot  { display: flex; justify-content: flex-start; margin: 2px 0; }
.bubble-row-user { display: flex; justify-content: flex-end;   margin: 2px 0; }
.msg-time { font-size: 10px; color: #999; margin-top: 2px; }
.bot-name { font-size: 11px; color: #075e54; font-weight: 600; margin-bottom: 2px; }
.stButton > button {
    border-radius: 22px !important;
    border: 1.5px solid #25d366 !important;
    color: #075e54 !important;
    background: #ffffff !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    transition: all .12s !important;
    width: 100% !important;
}
.stButton > button:hover { background: #e8fef0 !important; }
.hist-item {
    background: rgba(255,255,255,0.1);
    border-radius: 8px; padding: 8px 10px;
    margin: 4px 0;
    border-left: 3px solid #25d366;
    font-size: 12px;
}
.date-divider { text-align: center; margin: 10px 0; }
.date-divider span {
    background: rgba(225,245,254,.85);
    font-size: 11px; color: #777;
    padding: 3px 12px; border-radius: 8px;
}
.sim-badge {
    background: rgba(255,193,7,0.15);
    border: 1px solid rgba(255,193,7,0.5);
    border-radius: 8px; padding: 8px 12px;
    color: #856404; font-size: 12px; margin: 8px 0;
}
hr { border-color: rgba(0,0,0,0.08) !important; margin: 8px 0 !important; }
</style>
""", unsafe_allow_html=True)

# DATA LOADING
@st.cache_data(show_spinner=False)
def load_kb(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"❌ CSV haikupatikana: `{path}`")
        st.stop()
    required = {"faq_id","user_tier","category","language","question","answer"}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"❌ CSV imekosa columns: {missing}")
        st.stop()
    return df

def filter_kb(df, tier, category, language):
    mask = (df["user_tier"]==tier) & (df["category"]==category) & (df["language"]==language)
    return df[mask].reset_index(drop=True)

def build_context(filtered, max_pairs=8):
    rows = filtered.head(max_pairs)
    if rows.empty:
        return "(No specific entries found for this profile.)"
    return "\n\n".join(f"Q: {r['question']}\nA: {r['answer']}" for _,r in rows.iterrows())

def build_system_prompt(tier, category, language, context):
    lang_rule = (
        "You MUST respond ONLY in Kiswahili. Never switch to English."
        if language == "Kiswahili"
        else "You MUST respond ONLY in English."
    )
    return f"""You are Rafiki-Finance AI — a trusted expert AI financial advisor for Tanzania only.

USER PROFILE
• Tier     : {tier}
• Domain   : {category}
• Language : {language}

LANGUAGE RULE: {lang_rule}

DEPTH: {DEPTH_MAP.get(tier, "")}

SCOPE
1. Only answer about Tanzania's financial, banking, tax, insurance & business ecosystem.
2. Reference real institutions: TRA, BOT, BRELA, DSE, NMB, CRDB, NBC, M-Pesa, NSSF, NHIF, UTT-AMIS.
3. Cite laws when relevant: Income Tax Act Cap.332, Banking Act, VAT Act, Companies Act Cap.212.
4. Redirect if outside Tanzanian finance.
5. Suggest consulting a licensed professional for major decisions.

KNOWLEDGE BASE
{context}

FORMAT
• 2-4 short paragraphs.
• Bullet points for steps.
• Include USSD codes, TZS amounts, deadlines when relevant.
• End with "💡 Pro Tip:" (English) or "💡 Kidokezo:" (Kiswahili).

OUT OF SCOPE RULE — VERY IMPORTANT
If the user asks something completely unrelated to Tanzania's financial system
(e.g. sports, weather, cooking, politics, entertainment, general knowledge):
- If language is English, respond ONLY with this exact message:
  "I'm sorry, I'm unable to help with that. I'm specialized in Tanzania's financial topics only. Please ask me about savings, loans, taxes, mobile money, or business registration in Tanzania. 🇹🇿"
- If language is Kiswahili, respond ONLY with this exact message:
  "Samahani, siwezi kukusaidia na hilo. Mimi ni mtaalamu wa masuala ya fedha ya Tanzania pekee. Tafadhali niulize kuhusu akiba, mikopo, kodi, pesa ya simu, au usajili wa biashara Tanzania. 🇹🇿"

GIBBERISH / UNCLEAR INPUT RULE — VERY IMPORTANT
If the user types random letters, symbols, numbers, or words that make no sense
(e.g. "mamb", "asdfgh", "123abc", "??!!", "xyzxyz", "aaa", "jjjj"):
- If language is English, respond ONLY with this exact message:
  "I'm sorry, I didn't understand that. Could you please rephrase your question? I'm here to help with Tanzania's financial topics such as savings, loans, taxes, mobile money, and business registration. 🇹🇿"
- If language is Kiswahili, respond ONLY with this exact message:
  "Samahani, sijaelewa ulichoandika. Tafadhali uliza swali lako upya. Niko hapa kukusaidia kuhusu akiba, mikopo, kodi, pesa ya simu, na usajili wa biashara Tanzania. 🇹🇿"

GENERAL RULES
- Never attempt to answer out-of-scope or unclear questions.
- Never apologize more than once.
- Never guess what the user meant if input is completely unclear.
"""
def call_groq(messages, system_prompt, simulate):
    if simulate:
        q = messages[-1]["content"] if messages else "..."
        return (
            f"⚠️ **[SIMULATION MODE]**\n\nSwali lako: *\"{q}\"*\n\n"
            "Weka `GROQ_API_KEY` kwenye terminal:\n"
            "`set GROQ_API_KEY=gsk_xxxxxxxxxxxx`\n"
            "Kisha restart Streamlit."
        )
    try:
        from groq import Groq
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [{"role":"system","content":system_prompt}] + messages,
            temperature = 0.4,
            max_tokens  = 900,
        )
        return response.choices[0].message.content
    except ImportError:
        return "❌ Run: `pip install groq`"
    except Exception as e:
        return f"❌ Groq Error: {e}"

def speak_text(text, lang_code):
    safe = json.dumps(text.replace("**","").replace("*","").replace("💡",""))
    components.html(f"""
    <script>
    (function(){{
        try {{
            const s=window.speechSynthesis; s.cancel();
            const u=new SpeechSynthesisUtterance({safe});
            u.lang="{lang_code}"; u.rate=0.93; u.pitch=1.05;
            function go(){{
                const vs=s.getVoices();
                const v=vs.find(v=>v.lang.startsWith("{lang_code.split('-')[0]}"));
                if(v)u.voice=v; s.speak(u);
            }}
            s.getVoices().length?go():(s.onvoiceschanged=go);
        }}catch(e){{}}
    }})();
    </script>""", height=0)

def init_state():
    defaults = {
        "step":        "lang",
        "lang":        None,
        "tier":        None,
        "cat":         None,
        "hist":        [],
        "display":     [],
        "tts_enabled": False,
        "last_spoken": "",
        "sessions":    [],
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def bot_add(text):
    st.session_state.display.append({
        "role":"assistant","text":text,
        "time":datetime.now().strftime("%H:%M")
    })

def user_add(text):
    st.session_state.display.append({
        "role":"user","text":text,
        "time":datetime.now().strftime("%H:%M")
    })

def save_session():
    if not st.session_state.display:
        return
    title = next(
        (m["text"][:45]+"..." for m in st.session_state.display if m["role"]=="user"),
        "Chat session"
    )
    st.session_state.sessions.append({
        "title":   title,
        "tier":    st.session_state.tier or "—",
        "cat":     st.session_state.cat  or "—",
        "lang":    st.session_state.lang or "—",
        "date":    datetime.now().strftime("%d %b %Y %H:%M"),
        "display": st.session_state.display.copy(),
        "hist":    st.session_state.hist.copy(),
    })

def reset_chat():
    save_session()
    st.session_state.step        = "lang"
    st.session_state.lang        = None
    st.session_state.tier        = None
    st.session_state.cat         = None
    st.session_state.hist        = []
    st.session_state.display     = []
    st.session_state.last_spoken = ""

def load_session(idx):
    s = st.session_state.sessions[idx]
    st.session_state.step        = "chat"
    st.session_state.lang        = s["lang"]
    st.session_state.tier        = s["tier"]
    st.session_state.cat         = s["cat"]
    st.session_state.display     = s["display"].copy()
    st.session_state.hist        = s["hist"].copy()
    st.session_state.last_spoken = ""

# RENDER HELPERS
def render_header():
    sub = "Tanzania Financial Navigator"
    if st.session_state.tier and st.session_state.cat:
        sub = f"{st.session_state.cat}  ·  {st.session_state.tier}"
    av = (f'<img src="{BOT_AVATAR_URL}" style="width:44px;height:44px;border-radius:50%;object-fit:cover;">'
          if BOT_AVATAR_URL else f'<div class="header-avatar">{BOT_AVATAR}</div>')
    st.markdown(f"""
    <div class="header-card">
      {av}
      <div>
        <p class="header-name">Rafiki-Finance AI</p>
        <p class="header-sub"><span class="online-dot"></span>Online &nbsp;·&nbsp; {sub}</p>
      </div>
    </div>""", unsafe_allow_html=True)

def render_messages():
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    st.markdown('<div class="date-divider"><span>Today</span></div>', unsafe_allow_html=True)
    for m in st.session_state.display:
        txt = (m["text"]
       .replace("&","&amp;")
       .replace("<","&lt;")
       .replace(">","&gt;")
       .replace("\n","<br>")
       .strip())
        t = m.get("time","")
        if m["role"] == "assistant":
            st.markdown(f"""
            <div class="bubble-row-bot">
              <div>
                <div class="bot-name">Rafiki-Finance AI</div>
                <div class="bot-bubble">{txt}</div>
                <div class="msg-time">{t}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bubble-row-user">
              <div style="text-align:right">
                <div class="user-bubble">{txt}</div>
                <div class="msg-time">{t} ✓✓</div>
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

import re

def check_input(text: str, lang: str) -> str:
    isK = lang == "Kiswahili"
    t   = text.strip()

    if not t:
        return ""

    if len(t) < 3:
        return (
            "Samahani, sijaelewa ulichoandika. Tafadhali uliza swali lako upya. "
            "Niko hapa kukusaidia kuhusu fedha za Tanzania. 🇹🇿"
            if isK else
            "I'm sorry, I didn't understand that. Could you please rephrase "
            "your question? I'm here to help with Tanzania's financial topics. 🇹🇿"
        )

    if re.fullmatch(r'[\d\s\W]+', t):
        return (
            "Samahani, sijaelewa ulichoandika. Tafadhali uliza swali lako upya. 🇹🇿"
            if isK else
            "I'm sorry, I didn't understand that. Please type a proper question. 🇹🇿"
        )

    if re.fullmatch(r'(.)\1{3,}', t.replace(' ', '')):
        return (
            "Samahani, sijaelewa ulichoandika. Tafadhali uliza swali lako upya. 🇹🇿"
            if isK else
            "I'm sorry, I didn't understand that. Please rephrase your question. 🇹🇿"
        )

    letters_only = re.sub(r'[^a-zA-Z]', '', t.lower())
    if len(letters_only) >= 5:
        vowels = set('aeiou')
        vowel_count = sum(1 for c in letters_only if c in vowels)
        vowel_ratio = vowel_count / len(letters_only)
        if vowel_ratio < 0.10:
            return (
                "Samahani, sijaelewa ulichoandika. Tafadhali uliza swali lako upya. 🇹🇿"
                if isK else
                "I'm sorry, I didn't understand that. Please rephrase your question. 🇹🇿"
            )

    fillers_en = {'mmh','hmm','ok','okay','hi','hello','hey','yes','no','yeah','nope','lol','haha'}
    fillers_sw = {'mmh','sawa','ndiyo','hapana','haya','ehe','aa','oh','aaah','eeh','hmmm'}
    fillers    = fillers_en | fillers_sw
    if t.lower().strip('.,!?') in fillers:
        return (
            "Karibu! Niulize swali lolote kuhusu fedha za Tanzania — "
            "akiba, mikopo, kodi, pesa ya simu, au usajili wa biashara. 🇹🇿"
            if isK else
            "Hello! Feel free to ask me anything about Tanzania's financial system — "
            "savings, loans, taxes, mobile money, or business registration. 🇹🇿"
        )

    return ""


def main():
    init_state()
# MAIN
def main():
    init_state()
    df       = load_kb(CSV_PATH)
    simulate = not bool(GROQ_API_KEY)

    # render_sidebar()
    render_header()

    if simulate:
        st.markdown("""
        <div class="sim-badge">
        ⚠️ <b>Simulation Mode</b> — Weka <code>GROQ_API_KEY</code> kuwasha Llama 3.3 70B.<br>
        Terminal: <code>set GROQ_API_KEY=gsk_xxxxxxxxxxxx</code> kisha restart.
        </div>""", unsafe_allow_html=True)

    # Seed first message
    if not st.session_state.display:
        bot_add(
            "Rafiki-Finance AI 🇹🇿\n"
            "Your Tanzania Financial Navigator\n\n"
            "Welcome! Choose your language / Chagua lugha yako:"
        )

    render_messages()
    st.divider()

    # ── STEP: Language ──
    if st.session_state.step == "lang":
        c1, c2 = st.columns(2)
        if c1.button("🇬🇧 English", use_container_width=True):
            st.session_state.lang = "English"
            user_add("🇬🇧 English")
            bot_add("Great choice! Now tell me — who are you? Select your profile:")
            st.session_state.step = "tier"
            st.rerun()
        if c2.button("🇹🇿 Kiswahili", use_container_width=True):
            st.session_state.lang = "Kiswahili"
            user_add("🇹🇿 Kiswahili")
            bot_add("Asante! Sasa niambie — wewe ni nani? Chagua hali yako:")
            st.session_state.step = "tier"
            st.rerun()

    # ── STEP: Tier ──
    elif st.session_state.step == "tier":
        options = TIERS[st.session_state.lang]
        cols = st.columns(2)
        for i, (tid, icon, label) in enumerate(options):
            if cols[i%2].button(f"{icon} {label}", key=f"tier_{tid}", use_container_width=True):
                st.session_state.tier = tid
                user_add(f"{icon} {label}")
                bot_add(THANKS[tid][st.session_state.lang])
                isK = st.session_state.lang == "Kiswahili"
                bot_add(
                    "Sasa chagua eneo la fedha unalohitaji msaada:"
                    if isK else
                    "Now choose the financial area you need help with:"
                )
                st.session_state.step = "cat"
                st.rerun()

    # ── STEP: Category ──
    elif st.session_state.step == "cat":
        options = CATS[st.session_state.lang]
        cols = st.columns(2)
        for i, (cid, icon, label) in enumerate(options):
            if cols[i%2].button(f"{icon} {label}", key=f"cat_{cid}", use_container_width=True):
                st.session_state.cat = cid
                user_add(f"{icon} {label}")
                isK        = st.session_state.lang == "Kiswahili"
                tier_label = next(l for t,ic,l in TIERS[st.session_state.lang] if t==st.session_state.tier)
                bot_add(
                    f"Vizuri! Niko tayari kukusaidia kuhusu {label} kwa {tier_label}.\n\nNiulize swali lolote! 🇹🇿"
                    if isK else
                    f"Perfect! Ready to guide you on {label} for {tier_label}.\n\nAsk me anything! 🇹🇿"
                )
                st.session_state.step = "chat"
                st.rerun()

    # ── STEP: Chat ──
    elif st.session_state.step == "chat":
        isK        = st.session_state.lang == "Kiswahili"
        ph         = "Andika swali lako hapa..." if isK else "Type your financial question..."
        user_input = st.chat_input(ph)

        if user_input:
            user_add(user_input)

    # ── PRE-CHECK: gibberish na swali fupi sana ──
            rejection = check_input(user_input, st.session_state.lang)
            if rejection:
                bot_add(rejection)
                st.session_state.last_spoken = rejection
                st.rerun()

            st.session_state.hist.append({"role":"user","content":user_input})
            filtered   = filter_kb(df, st.session_state.tier, st.session_state.cat, st.session_state.lang)
            context    = build_context(filtered)
            sys_prompt = build_system_prompt(
                st.session_state.tier,
                st.session_state.cat,
                st.session_state.lang,
                context
            )

            spin_msg = "Rafiki-Finance AI inafikiria..." if isK else "Thinking..."
            with st.spinner(spin_msg):
                reply = call_groq(st.session_state.hist, sys_prompt, simulate)

            bot_add(reply)
            st.session_state.hist.append({"role":"assistant","content":reply})
            st.session_state.last_spoken = reply
            st.rerun()

   

if __name__ == "__main__":
    main()
