from pathlib import Path



def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    return path.parent

PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / 'results'
LANGUAGES_DIR = RESULTS_DIR / 'languages'