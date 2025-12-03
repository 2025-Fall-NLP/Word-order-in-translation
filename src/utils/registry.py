"""Generic registry factory for extensible component registration."""

from typing import Callable, Dict, List, Type, TypeVar

T = TypeVar("T")


def create_registry(name: str):
    """Create a registry. Returns (register_decorator, get_fn, list_fn)."""
    _registry: Dict[str, Type] = {}

    def register(type_name: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            if type_name in _registry:
                raise ValueError(f"{name} '{type_name}' already registered")
            _registry[type_name] = cls
            return cls
        return decorator

    def get(type_name: str) -> Type:
        if type_name not in _registry:
            raise KeyError(f"Unknown {name}: '{type_name}'. Available: {list(_registry.keys())}")
        return _registry[type_name]

    def list_registered() -> List[str]:
        return list(_registry.keys())

    return register, get, list_registered
