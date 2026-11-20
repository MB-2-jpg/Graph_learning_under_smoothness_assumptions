import sys
from pathlib import Path
from typing import List

__all__ = [
    "ROOT", "SRC", "COMMON", "ARTIFICIAL_SRC", "REEL_SRC",
    "DATA", "DATA_REEL", "OUTPUTS", "OUTPUTS_ARTIFICIAL", "OUTPUTS_REEL",
    "REPORTS", "REPORTS_ARTIFICIAL", "REPORTS_REEL",
    "data_raw", "data_interim", "ensure", "bootstrap",
]

ROOT: Path = Path(__file__).resolve().parent.parent.parent

SRC: Path = ROOT / "src"
COMMON: Path = SRC / "common"
ARTIFICIAL_SRC: Path = SRC / "artificial_graphs"
REEL_SRC: Path = SRC / "reel_graphs"

DATA: Path = ROOT / "data"
DATA_REEL: Path = DATA / "reel_graphs"

OUTPUTS: Path = ROOT / "outputs"
OUTPUTS_ARTIFICIAL: Path = OUTPUTS / "artificial_graphs"
OUTPUTS_REEL: Path = OUTPUTS / "reel_graphs"

REPORTS: Path = ROOT / "reports"
REPORTS_ARTIFICIAL: Path = REPORTS / "experiments_artificial_graphs"
REPORTS_REEL: Path = REPORTS / "experiments_reel_graphs_data"


def data_raw(dataset: str) -> Path:
    return DATA_REEL / dataset / "data_raw"


def data_interim(dataset: str) -> Path:
    return DATA_REEL / dataset / "data_interim"


def ensure(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def bootstrap(extra: str = "") -> List[str]:
    wanted = [COMMON]
    for name in extra.split():
        wanted.append(SRC / name)
    added = []
    for directory in wanted:
        text = str(directory)
        if directory.is_dir() and text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
    return added
