"""shad0w: compile the decisions your model already makes into a certified table that answers in microseconds.

Runtime needs only numpy (plus the bundled C core when present). No network, no GPU, no telemetry; the only
network call is the one you ask for (`llm_teacher` / `decision(..., llm=...)` calling your own LLM).
"""
from .api import Model, compile, load  # noqa: F401
from .cascade import Decision, Shadow, cascade, decide, decision, explain  # noqa: F401
from .config import Secret, Settings, configure  # noqa: F401
from .config import resolve as settings
from .llm import LLMTeacher, TeacherError, llm_teacher  # noqa: F401
from .observe import Metrics  # noqa: F401
from .shadow import certify_bundle, read_certificate, shadow_compile  # noqa: F401

__version__ = "0.3.3"
__all__ = ["Model", "compile", "load", "Shadow", "Decision", "cascade", "decision", "decide", "explain", "llm_teacher",
           "LLMTeacher", "TeacherError", "Metrics", "Settings", "settings", "configure", "Secret", "shadow_compile", "certify_bundle",
           "read_certificate", "__version__"]
