"""The bash the tests run, resolved once by path.

On Windows, subprocess resolves a bare program name through CreateProcess,
which searches System32 before PATH. With WSL installed, System32\\bash.exe
is WSL's launcher, so a bare "bash" starts a Linux bash that reads the
backslashes of a Windows script path as escapes and finds nothing, while
shutil.which("bash") returns Git Bash from PATH (#308). Every test that
starts bash takes the name from here; tests/test_bash_pinned.py fails when
a bare one comes back.

Resolved at import, not per call: some tests hand bash a PATH with no
interpreter on it, and an empty PATH would leave bash itself unfindable.

Importable as ``from _bash import BASH``: tests/ has no __init__.py, so
pytest inserts it on sys.path when it collects the modules there.
"""

from __future__ import annotations

import shutil

BASH = shutil.which("bash") or "bash"
