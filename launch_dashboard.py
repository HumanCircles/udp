"""Entry point — run this to open the ICP Filter Dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.dashboard import launch_dashboard
launch_dashboard()
