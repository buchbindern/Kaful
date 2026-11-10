"""Twin backends. Library adapters live in subpackages; `base` defines the interface."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaful.backends.base import TwinBackend


def load_backend(name: str) -> "TwinBackend":
    """The adapter `name`, imported on first use: importing the pipeline never loads a backend library."""
    if name == "progpy":
        from kaful.backends.progpy.backend import ProgPyBackend

        return ProgPyBackend()
    raise ValueError(f"unknown twin backend {name!r}")
