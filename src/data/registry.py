"""Registry for dataset loaders."""

from src.utils.registry import create_registry

register_dataset, get_dataset, list_datasets = create_registry("dataset")

