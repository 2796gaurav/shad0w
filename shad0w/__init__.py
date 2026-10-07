"""shad0w: compile the decisions your model already makes into a certified table that answers in microseconds.

Runtime needs only numpy (plus the bundled C core when present). No network, no GPU, no telemetry.
"""
from .api import Model, compile, load  # noqa: F401
from .cascade import Decision, Shadow, cascade  # noqa: F401
from .shadow import certify_bundle, read_certificate, shadow_compile  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Model", "compile", "load", "Shadow", "Decision", "cascade", "shadow_compile", "certify_bundle",
           "read_certificate", "__version__"]
