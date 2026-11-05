"""
Forwarding entrypoint for Streamlit Cloud deployments where Main file path is set to 'main/app.py'.
"""
import os
import sys

# Ensure repository root is on Python module search path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

root_app = os.path.join(REPO_ROOT, "app.py")
with open(root_app, "r", encoding="utf-8") as f:
    exec(compile(f.read(), root_app, "exec"), globals())

