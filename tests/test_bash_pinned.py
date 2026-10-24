"""Every bash the suite starts is the one shutil.which finds, never a bare name (#308).

On a Windows machine with WSL installed, subprocess.run(["bash", ...]) does
not start Git Bash. CreateProcess searches System32 before PATH, and
System32\\bash.exe is WSL's launcher, so the script path (a Windows path with
backslashes) reaches a Linux bash that reads them as escapes and reports
"No such file or directory". shutil.which("bash") returns Git Bash on the same
machine. 45 tests failed this way and passed once bash was resolved by path.

The fix is one resolver, tests/_bash.py, and these fences keep it that way:
a call whose argv starts with the literal "bash", a command string handed to
a subprocess environment that starts with "bash ", and a second copy of the
resolver in a test module each fail here, with the file and line named.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

from _bash import BASH

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "tests" / "_bash.py"
THIS = Path(__file__).resolve()
# Scripts the suite runs the same way CI does, so a bare bash there fails on
# the same machines. They resolve locally rather than importing from tests/.
SCRIPTS = (REPO_ROOT / "scripts" / "conformance_report.py",)


def _sources() -> list[Path]:
    return sorted(REPO_ROOT.glob("tests/*.py")) + list(SCRIPTS)


def _leading_text(node: ast.expr) -> str | None:
    """The literal a string starts with: the whole constant, or the constant
    part before the first placeholder of an f-string. None for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
    return None


def test_helper_resolves_bash_by_path():
    found = shutil.which("bash")
    if found is None:
        assert BASH == "bash"
    else:
        assert BASH == found
        assert Path(BASH).is_absolute()


def test_no_call_starts_a_bare_bash():
    """argv[0] is BASH, not "bash". Expected argv values in the external-command
    split test are list displays inside a dict, not call arguments, so they
    are left alone: the fence reads what is handed to a call."""
    bare = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            argv = node.args[0]
            if isinstance(argv, ast.List) and argv.elts and _leading_text(argv.elts[0]) == "bash":
                bare.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not bare, "argv starts with a bare 'bash'; use BASH from tests/_bash.py:\n  " + "\n  ".join(bare)


def test_no_command_string_starts_a_bare_bash():
    """RETRIEVAL_EVAL_EXTERNAL_CMD and anything like it: a command string given
    to a call as a keyword value (dict(os.environ, X="bash ...")) reaches a
    subprocess argv through the callee and resolves the same wrong way."""
    bare = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                text = _leading_text(kw.value)
                if text is not None and text.startswith("bash "):
                    bare.append(f"{path.relative_to(REPO_ROOT)}:{kw.value.lineno}")
    assert not bare, "command string starts with a bare 'bash'; quote BASH instead:\n  " + "\n  ".join(bare)


def test_only_the_helper_resolves_bash():
    """A machine-specific fix should land once. A second `shutil.which("bash")`
    in a test module is the copy that drifts (see tests/test_osb_env.py for
    what four copies of one resolver cost)."""
    copies = []
    for path in _sources():
        if path in (HELPER, THIS) or path in SCRIPTS:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if 'which("bash")' in line and not line.lstrip().startswith("#"):
                copies.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not copies, "resolve bash in tests/_bash.py only:\n  " + "\n  ".join(copies)
