from pathlib import Path


def find_project_root(current_path: Path) -> Path:
    """Finds root dir by looking for pyproject.toml"""
    while current_path.parent != current_path:
        if (current_path / 'pyproject.toml').exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("no pyproject.toml found")


def is_valid_model(config, model_id, service):
    model_config = config['model-config']
    return (
            model_id in model_config and
            service in model_config[model_id]['access']
    )
