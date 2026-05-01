"""Save-As download buttons backed by hidden st.download_button + showSaveFilePicker."""
from __future__ import annotations

from typing import Callable

import streamlit as st

# ---- Inline SVG icons (currentColor lets the CSS accent color them) -------
ICON_CSV = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
    '<polyline points="14 2 14 8 20 8"/>'
    '<path d="M8 13h2M8 17h2M14 13h2M14 17h2"/></svg>'
)
ICON_PARQUET = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="4" width="18" height="4" rx="1"/>'
    '<rect x="3" y="10" width="18" height="4" rx="1"/>'
    '<rect x="3" y="16" width="18" height="4" rx="1"/></svg>'
)
ICON_ARCHIVE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 8v13H3V8"/><path d="M1 3h22v5H1z"/>'
    '<line x1="10" y1="12" x2="14" y2="12"/></svg>'
)


def save_as_button(
    label: str,
    sublabel: str,
    icon_svg: str,
    accent: str,
    payload_fn: Callable[[], bytes],
    suggested_name: str,
    mime: str,
    ext: str,
    accept_desc: str,
    component_key: str,
) -> None:
    """Render a styled download button.

    A hidden native `st.download_button` (with a lazy callable) sits behind the
    visual HTML button. No bytes are built at render time — Streamlit only
    calls `payload_fn()` when the user actually clicks.
    """
    anchor_id = f"__dl_anchor_{component_key}__"

    # 1. Invisible anchor marker so the iframe JS can locate the native button.
    st.markdown(
        f'<span id="{anchor_id}" style="display:none"></span>',
        unsafe_allow_html=True,
    )
    # 2. Hidden native download button — bytes built lazily on click only.
    st.download_button(
        label="dl",
        data=payload_fn,
        file_name=suggested_name,
        mime=mime,
        key=f"_hb_{component_key}",
        use_container_width=False,
    )
    # 3. CSS to hide the anchor marker row and the native button row.
    st.markdown(
        f"""<style>
  div[data-testid="stMarkdown"]:has(#{anchor_id}),
  div[data-testid="stMarkdown"]:has(#{anchor_id}) + div,
  div[data-testid="stElementContainer"]:has(#{anchor_id}),
  div[data-testid="stElementContainer"]:has(#{anchor_id}) + div {{
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
  }}
</style>""",
        unsafe_allow_html=True,
    )

    # 4. Visual HTML button — identical design, JS clicks the hidden button.
    height = 116
    elem_id = f"sa_{component_key}"
    html_doc = f"""
<!doctype html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ overflow: visible !important; }}
  body {{
    margin: 0; padding: 8px 4px 16px 4px;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: transparent;
  }}
  .save-btn {{
    width: 100%;
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.65rem 0.85rem;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.25);
    background: linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,41,59,0.85));
    color: #e2e8f0;
    font-size: 0.92rem; font-weight: 600;
    cursor: pointer;
    transition: transform 0.12s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    text-align: left;
    backdrop-filter: blur(6px);
  }}
  .save-btn:hover {{
    transform: translateY(-1px);
    border-color: {accent};
    box-shadow: 0 8px 24px -10px {accent}, 0 0 0 1px {accent}33 inset;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95));
  }}
  .save-btn:active {{ transform: translateY(0); }}
  .save-btn:focus-visible {{ outline: 2px solid {accent}; outline-offset: 2px; }}
  .icon {{
    flex: 0 0 36px; height: 36px; width: 36px;
    display: grid; place-items: center;
    border-radius: 9px;
    background: {accent}22;
    color: {accent};
  }}
  .icon svg {{ width: 18px; height: 18px; }}
  .text {{ display: flex; flex-direction: column; line-height: 1.15; min-width: 0; }}
  .text .label {{ font-size: 0.95rem; }}
  .text .sub {{ font-size: 0.72rem; color: #94a3b8; font-weight: 500; margin-top: 2px; }}
  .status {{
    font-size: 0.74rem; color: #94a3b8;
    margin-top: 0.35rem; padding-left: 0.2rem; min-height: 1em;
  }}
  .ok {{ color: #34d399; }}
  .err {{ color: #f87171; }}
</style></head>
<body>
  <button id="{elem_id}-btn" class="save-btn" type="button" aria-label="{label}">
    <span class="icon">{icon_svg}</span>
    <span class="text">
      <span class="label">{label}</span>
      <span class="sub">{sublabel}</span>
    </span>
  </button>
  <div class="status" id="{elem_id}-msg"></div>
<script>
(function() {{
  const anchorId = "{anchor_id}";
  const suggestedName = {suggested_name!r};
  const mime = {mime!r};
  const ext = {ext!r};
  const acceptDesc = {accept_desc!r};
  const btn = document.getElementById("{elem_id}-btn");
  const msg = document.getElementById("{elem_id}-msg");

  function findDlContainer(retries, cb) {{
    try {{
      const doc = window.parent.document;
      const marker = doc.getElementById(anchorId);
      if (marker) {{
        let scope = marker.closest('[data-testid="stColumn"]');
        if (!scope) {{
          let el = marker.parentElement;
          for (let i = 0; i < 20 && el; i++, el = el.parentElement) {{
            if (el.querySelector('[data-testid="stDownloadButton"]')) {{ scope = el; break; }}
          }}
        }}
        if (scope) {{
          const container = scope.querySelector('[data-testid="stDownloadButton"]');
          if (container) {{ cb(container); return; }}
        }}
      }}
    }} catch(e) {{}}
    if (retries > 0) setTimeout(function() {{ findDlContainer(retries - 1, cb); }}, 150);
    else {{ msg.textContent = "Download failed \u2014 try refreshing."; msg.className = "status err"; }}
  }}

  async function doSaveAs(container) {{
    const a = container.querySelector('a[href]');
    if (a && a.href) {{
      msg.textContent = "Preparing\u2026";
      msg.className = "status ok";
      try {{
        const resp = await fetch(a.href);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const blob = await resp.blob();

        if (window.showSaveFilePicker) {{
          const handle = await window.showSaveFilePicker({{
            suggestedName: suggestedName,
            types: [{{ description: acceptDesc, accept: {{ [mime]: [ext] }} }}],
          }});
          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
          msg.textContent = "\u2713 Saved: " + handle.name;
          msg.className = "status ok";
        }} else {{
          const url = URL.createObjectURL(blob);
          const dl = document.createElement("a");
          dl.href = url; dl.download = suggestedName;
          document.body.appendChild(dl); dl.click(); dl.remove();
          URL.revokeObjectURL(url);
          msg.textContent = "Downloaded (Save As\u2026 requires Chrome/Edge).";
          msg.className = "status ok";
        }}
      }} catch(e) {{
        if (e && e.name === "AbortError") {{
          msg.textContent = "Cancelled.";
          msg.className = "status";
        }} else {{
          msg.textContent = "Error: " + (e.message || e);
          msg.className = "status err";
        }}
      }}
    }} else {{
      const nativeBtn = container.querySelector('button');
      if (nativeBtn) {{
        nativeBtn.click();
        msg.textContent = "Downloading\u2026";
        msg.className = "status ok";
        setTimeout(function() {{ msg.textContent = ""; }}, 2500);
      }} else {{
        msg.textContent = "Download failed \u2014 try refreshing.";
        msg.className = "status err";
      }}
    }}
  }}

  btn.addEventListener("click", function() {{
    findDlContainer(5, doSaveAs);
  }});
}})();
</script>
</body></html>
"""
    st.components.v1.html(html_doc, height=height)
