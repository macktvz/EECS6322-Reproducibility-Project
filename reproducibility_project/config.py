from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data"
DATA_PATH.mkdir(exist_ok=True)