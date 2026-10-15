import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"))
