"""
Chatbot.py  —  Streamlit Page Entry Point
==========================================
Place this file at:   AI-Pediatric-Pneumonia-Detector/apppy/pages/Chatbot.py

It imports and calls the render function from agentt/layer4_5.py.
No logic lives here — this is purely a routing shim so Streamlit
picks it up as a page in the multi-page app.

Navigation: add a button in app.py pointing to this page:
    if st.button("Clinical Assistant"):
        st.switch_page("pages/Chatbot.py")
"""

import sys
from pathlib import Path

# Make agentt/ importable from the pages/ directory
# pages/ → apppy/ → project root → agentt/
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from agentt.layer4_5 import render_chatbot_page

render_chatbot_page()