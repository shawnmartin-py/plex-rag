from nicegui import ui

SIDEBAR_WIDTH_PX = 288

# Confirmed via a throwaway NiceGUI spike page + Playwright DOM inspection:
# ui.add_head_html's <style> tag is NOT wrapped in any CSS @layer, and
# unlayered rules always beat layered rules regardless of specificity —
# NiceGUI 3.x's own styles (Quasar, Tailwind utilities, etc.) all live inside
# named @layers (see the `@layer theme, base, quasar, ...` statement NiceGUI
# emits). So this stylesheet reliably overrides built-in styling without
# needing `!important` or explicit @layer wrapping; `!important` is kept
# anyway to mirror the original Streamlit stylesheet's defensiveness.
#
# Design direction: "Glass × Cinematic" — frosted-glass chrome (sidebar,
# input bar) floating over a cool ambient field, with each recommendation
# card backed by a blurred copy of its own poster behind a left-to-right
# dark scrim. Three text tiers: #F5F5F7 primary, #A1A1A6 secondary,
# #6E6E73 tertiary; #0A84FF is reserved for interactive elements.
DARK_CSS = """
<style>
/* ── Design tokens ────────────────────────────────────────────────── */
:root {
    --q-primary: #0A84FF;
    --plex-ink: #F5F5F7;
    --plex-ink-2: #A1A1A6;
    --plex-ink-3: #6E6E73;
    --plex-accent: #0A84FF;
    --plex-glass: rgba(28, 28, 32, 0.55);
    --plex-glass-border: rgba(255, 255, 255, 0.12);
}

/* ── Ground: near-black with a cool, neutral ambient field ───────── */
body {
    background:
        radial-gradient(520px 420px at 78% 6%, rgba(44,54,74,.34), transparent 70%),
        radial-gradient(560px 480px at 30% 88%, rgba(38,72,94,.36), transparent 70%),
        radial-gradient(420px 380px at 8% 30%, rgba(10,132,255,.12), transparent 70%),
        #050507 !important;
    background-attachment: fixed !important;
    color: var(--plex-ink) !important;
}
.q-layout, .q-page {
    background: transparent !important;
}

/* ── System font ─────────────────────────────────────────────────── */
body, button, input, textarea {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Headings ────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    letter-spacing: -0.03em !important;
    font-weight: 600 !important;
}

/* ── Sidebar: floating frosted panel ─────────────────────────────── */
/* NiceGUI applies our classes to the q-drawer__content div, so
   .plex-sidebar IS the content element; the flat dark background lives on
   its .q-drawer parent (aside), which must go transparent for the floating
   panel + backdrop blur to read. */
.q-drawer {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.plex-sidebar {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    width: calc(100% - 24px) !important;
    height: calc(100% - 24px) !important;
    margin: 12px !important;
    padding: 18px 16px !important;
    border-radius: 20px !important;
    background: var(--plex-glass) !important;
    backdrop-filter: blur(28px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(28px) saturate(1.5) !important;
    border: 1px solid var(--plex-glass-border) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 12px 40px rgba(0, 0, 0, 0.5) !important;
}
.plex-sb-head {
    flex-wrap: nowrap !important;
    gap: 10px !important;
    margin-bottom: 18px !important;
}
.plex-app-mark {
    width: 30px !important;
    height: 30px !important;
    border-radius: 8px !important;
    flex: none !important;
    background: linear-gradient(140deg, #1c3a5e, #0A84FF) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.plex-app-mark .q-icon {
    color: #ffffff !important;
    font-size: 20px !important;
}
.plex-sb-name {
    flex: 1 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    letter-spacing: -0.022em !important;
    color: var(--plex-ink) !important;
}
.plex-icon-btn {
    color: var(--plex-ink-2) !important;
    border-radius: 9px !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
.plex-icon-btn svg {
    display: block !important;
}
.plex-icon-btn:hover {
    background: rgba(255, 255, 255, 0.09) !important;
    color: var(--plex-ink) !important;
}
.plex-float-toggle {
    position: fixed !important;
    top: 14px !important;
    left: 18px !important;
    z-index: 2500 !important;
    background: rgba(28, 28, 32, 0.6) !important;
    backdrop-filter: blur(20px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.5) !important;
    border: 1px solid var(--plex-glass-border) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

/* ── New conversation button ─────────────────────────────────────── */
.plex-new-conv-btn {
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: var(--plex-ink) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    letter-spacing: -0.1px !important;
    text-transform: none !important;
    padding: 7px 16px !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
}
.plex-new-conv-btn:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}
.plex-new-conv-btn .q-icon {
    font-size: 16px !important;
}

/* ── Recent conversations (static placeholder for now) ───────────── */
.plex-sec-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--plex-ink-3) !important;
    margin: 18px 4px 4px !important;
}
.plex-conv {
    font-size: 13.5px !important;
    color: var(--plex-ink-2) !important;
    padding: 8px 12px !important;
    border-radius: 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
}
.plex-conv:hover {
    background: rgba(255, 255, 255, 0.06) !important;
}
.plex-conv-active {
    background: rgba(255, 255, 255, 0.11) !important;
    color: var(--plex-ink) !important;
    font-weight: 500 !important;
}

/* ── Sidebar footer (spoiler toggle pinned to the bottom) ────────── */
.plex-sb-bottom {
    margin-top: auto !important;
}
.plex-sidebar .q-toggle {
    color: var(--plex-ink-2) !important;
    font-size: 13.5px !important;
}

/* ── Main column ─────────────────────────────────────────────────── */
.plex-main {
    background: transparent !important;
    max-width: 920px !important;
    margin: 0 auto !important;
    padding-top: 2.25rem !important;
}

/* ── Chat messages ───────────────────────────────────────────────── */
.plex-msg-row {
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    margin-bottom: 2px !important;
}
.plex-msg-user {
    margin-left: auto !important;
    max-width: 60% !important;
    background: var(--plex-accent) !important;
    color: #ffffff !important;
    border-radius: 18px !important;
    border-bottom-right-radius: 5px !important;
    padding: 9px 15px !important;
    font-size: 14.5px !important;
    letter-spacing: -0.01em !important;
}
.plex-msg-user p {
    margin: 0 !important;
    color: #ffffff !important;
}
.plex-msg-prose {
    color: var(--plex-ink-2) !important;
    font-size: 14.5px !important;
    max-width: 68ch !important;
}
.plex-msg-prose p {
    color: var(--plex-ink-2) !important;
}

/* ── Movie cards: blurred-poster backdrop + scrim ────────────────── */
.plex-card {
    position: relative !important;
    overflow: hidden !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.09),
                0 18px 50px rgba(0, 0, 0, 0.55) !important;
    padding: 26px 30px !important;
    margin: 6px 0 16px !important;
    background: #101012 !important; /* fallback when there is no poster */
}
.plex-card-bg {
    position: absolute !important;
    inset: -40px !important;
    background-size: cover !important;
    background-position: center !important;
    filter: blur(46px) saturate(1.4) !important;
    transform: scale(1.25) !important;
}
.plex-card-scrim {
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(92deg,
        rgba(0, 0, 0, 0.16) 0%,
        rgba(4, 4, 6, 0.78) 34%,
        rgba(4, 4, 6, 0.94) 60%) !important;
}
.plex-card-inner {
    position: relative !important;
    flex-wrap: nowrap !important;
    gap: 30px !important;
}
/* A fixed 2:3 poster box sized to a typical card's text height (three
   labeled bullets). Letting the poster track the card height instead
   creates circular sizing — the poster then drives the card taller than
   its text. object-fit: cover absorbs thumbs that aren't exactly 2:3. */
.plex-poster-col {
    flex: 0 0 230px !important;
}
.plex-poster {
    width: 100% !important;
    height: auto !important;
    aspect-ratio: 2 / 3 !important;
    object-fit: cover !important;
    border-radius: 12px !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.65) !important;
    display: block !important;
}
.plex-text-col {
    flex: 1 !important;
    min-width: 0 !important;
    gap: 0 !important;
}
.plex-card-eyebrow {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    opacity: 0.85;
    margin-bottom: 8px !important;
}
.plex-title-row {
    flex-wrap: nowrap !important;
    gap: 8px !important;
    margin-bottom: 4px !important;
}
.plex-card-title {
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: -0.028em !important;
    color: var(--plex-ink) !important;
    line-height: 1.2 !important;
}
.plex-card-year {
    font-size: 20px !important;
    font-weight: 400 !important;
    color: var(--plex-ink-2) !important;
}
.plex-card-meta {
    flex-wrap: nowrap !important;
    gap: 9px !important;
    margin-bottom: 12px !important;
}
.plex-badge {
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    font-variant-numeric: tabular-nums !important;
    background: rgba(255, 255, 255, 0.16) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
    color: var(--plex-ink) !important;
}
.plex-genres {
    font-size: 13px !important;
    color: var(--plex-ink-2) !important;
}
.plex-platform-badge {
    height: 18px !important;
    width: auto !important;
    display: block !important;
}

/* LLM prose inside a card: bold run-in labels become block labels,
   bullets disappear — the structure reads as designed UI, not markdown. */
.plex-card-body {
    color: #D1D1D6 !important;
    font-size: 14.5px !important;
    max-width: 65ch !important;
}
.plex-card-body p, .plex-card-body li {
    color: #D1D1D6 !important;
}
.plex-card-body ul {
    list-style: none !important;
    padding-left: 0 !important;
    margin: 0 !important;
}
.plex-card-body li {
    margin-bottom: 12px !important;
}
.plex-card-body li > strong:first-child,
.plex-card-body p > strong:first-child {
    display: block !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 1px !important;
}

/* ── Chat input: frosted bar, pinned while the transcript scrolls ── */
.plex-input-row {
    position: sticky !important;
    bottom: 0 !important;
    z-index: 10 !important;
    padding: 12px 0 20px !important;
    background: linear-gradient(to top, rgba(5, 5, 7, .75) 60%, transparent) !important;
}
.plex-chat-input .q-field__control {
    background: rgba(28, 28, 32, 0.6) !important;
    backdrop-filter: blur(28px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(28px) saturate(1.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 24px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 10px 30px rgba(0, 0, 0, 0.4) !important;
    padding: 0 8px 0 18px !important;
}
.plex-chat-input.q-field--focused .q-field__control {
    border-color: rgba(10, 132, 255, 0.55) !important;
}
.plex-chat-input .q-field__control:before,
.plex-chat-input .q-field__control:after {
    border: none !important;
}
.plex-chat-input input {
    background: transparent !important;
    color: var(--plex-ink) !important;
    font-size: 15px !important;
}
.plex-send-icon {
    width: 30px !important;
    height: 30px !important;
    border-radius: 50% !important;
    background: var(--plex-accent) !important;
    color: #ffffff !important;
    font-size: 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.15s ease !important;
}
.plex-send-icon:hover {
    background: #3395FF !important;
}
</style>
"""


def apply_theme() -> None:
    ui.dark_mode().enable()
    ui.add_head_html(DARK_CSS)
