from pathlib import Path



def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    return path.parent

def get_figures_folder():
    return get_project_root().parent / 'Write Up' / 'Dissertation' / 'figures'


PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / 'results'
LANGUAGES_DIR = RESULTS_DIR / 'languages'


FIGURES_DIR = get_figures_folder()