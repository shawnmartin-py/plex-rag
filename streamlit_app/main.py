import streamlit as st

from streamlit_app.components import render_recommendations
from streamlit_app.init import build_service

st.set_page_config(page_title="Plex Movie Assistant", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
/* ── System font ─────────────────────────────────────────────────── */
html, body, [class*="st-"], button, input, textarea {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Headings ────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3 {
    letter-spacing: -0.03em !important;
    font-weight: 600 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: #080808 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}
[data-testid="stSidebar"] h1 {
    font-size: 17px !important;
    font-weight: 600 !important;
    letter-spacing: -0.4px !important;
}

/* ── New conversation button ─────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #F5F5F7 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: -0.1px !important;
    padding: 6px 16px !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
}

/* ── Movie card rows ─────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    background: #111111 !important;
    border-radius: 18px !important;
    padding: 20px 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    margin-bottom: 4px !important;
}

/* ── Poster images ───────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7) !important;
    display: block !important;
}

/* ── Captions (rating, genres) ───────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #8E8E93 !important;
    font-size: 13px !important;
    letter-spacing: 0.1px !important;
    margin-top: 2px !important;
}

/* ── Dividers ────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.07) !important;
    margin: 16px 0 !important;
}

/* ── Chat messages ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    gap: 12px !important;
}

/* ── Chat input ──────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #111111 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(10, 132, 255, 0.55) !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #F5F5F7 !important;
    font-size: 15px !important;
}
/* ── Preserve Material Icon font for Streamlit icons ──────────────── */
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}

/* ── Main content padding ────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    padding-top: 3.5rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("🎬 Plex Movie Assistant")
    st.divider()
    spoiler_free = st.toggle("Spoiler-free mode", value=False)
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        if "service" in st.session_state:
            st.session_state.service.reset_history()
        st.rerun()

# --- Initialize service (cached per spoiler_free setting) ---
with st.spinner("Loading movie library..."):
    service, sql_repo = build_service(spoiler_free=spoiler_free)
st.session_state.service = service

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat history ---
for msg in st.session_state.messages:
    avatar = "🎬" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_recommendations(msg["content"], sql_repo)
        else:
            st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask for a movie recommendation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎬"):
        with st.spinner("Finding recommendations..."):
            answer, _ = service.chat_with_items(prompt, sql_repo)
        render_recommendations(answer, sql_repo)

    st.session_state.messages.append({"role": "assistant", "content": answer})
