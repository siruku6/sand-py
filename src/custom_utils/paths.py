import os
from pathlib import Path

APP_ROOT_DIR: Path = Path(os.path.abspath(__file__)).parents[2]

CONFIG_DIR: Path = APP_ROOT_DIR / "configs"

DATA_PATH: Path = APP_ROOT_DIR / "data"

TEST_FIXTURES_DIR: Path = APP_ROOT_DIR / "tests" / "fixtures"

OUTPUT_PATH: Path = APP_ROOT_DIR / "outputs"
