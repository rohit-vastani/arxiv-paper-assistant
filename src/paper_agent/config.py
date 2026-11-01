from dataclasses import dataclass
import os
from pathlib import Path
main_path = Path(os.path.dirname(__file__)).resolve().parents[1] / "data"


@dataclass
class ResourcePath:
    MAIN_DIR: str = str(main_path)
    pdf_dir: str = str(main_path / "pdfs")
    chroma_dir: str = str(main_path / "chroma")


if __name__ == "__main__":

    # Check all the paths
    print(main_path)
    PATHS = ResourcePath()
    print(PATHS.MAIN_DIR)
    print(PATHS.pdf_dir)
    print(PATHS.chroma_dir)