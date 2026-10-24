# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The environment-variable registry and the argument spellings of the
engine tree being launched.

Read statically (by AST) from ``<tree>/vllm/envs.py`` and, for an sm80
tree, ``<tree>/vllm/sm80/envs.py``. Never imports ``vllm`` (its package
import pulls in torch) and never uses the launcher's own commit's tables:
the names, retired-name texts and code defaults always come from the tree
that will run. The long option spellings come the same way from the files
that declare the server's arguments (``ARG_SOURCES``), by the two ways the
engine registers them: every literal ``--name`` passed to ``add_argument``
(with the aliases of a call that names several), and the options a
frontend class generates from its dataclass fields (see
``_generated_options``). The negative ``--no-X`` forms of boolean options
are not listed; a ``no-`` spelling the tree does not declare is read as
the flag it negates.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field

from canon import sha256_file
from errors import Refusal

CAPABILITIES = ("sm80", "plain-upstream")
ARG_SOURCES = ("vllm/engine/arg_utils.py", "vllm/entrypoints/launchers/cli_args.py")


@dataclass
class Registry:
    tree: str
    capability: str
    upstream: set[str]
    fork: set[str]
    retired: dict[str, str]
    defaults: dict[str, str | None] = field(default_factory=dict)
    envs_sha256: str = ""
    sm80_envs_sha256: str | None = None

    @property
    def names(self) -> set[str]:
        return self.upstream | self.fork

    def check(self, name: str) -> str | None:
        """None if the name is registered, else the refusal text."""
        if name in self.retired:
            return f"{name} is retired in this engine tree: {self.retired[name]}"
        if name in self.names:
            return None
        if name.upper() in self.names or name.upper() in self.retired:
            return f"{name} is not a registered name (names are case-sensitive; did you mean {name.upper()}?)"
        return f"{name} is not registered in this engine tree's vllm/envs.py or vllm/sm80/envs.py"

    def describe(self) -> dict:
        return {
            "source": "arm-tree",
            "tree": self.tree,
            "envs_sha256": self.envs_sha256,
            "sm80_envs_sha256": self.sm80_envs_sha256,
            "names": len(self.names),
            "retired": len(self.retired),
        }


def _dict_node(tree: ast.Module, var: str) -> ast.Dict | None:
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target == var and isinstance(value, ast.Dict):
            return value
    return None


def _keys(node: ast.Dict, where: str) -> list[str]:
    out = []
    for k in node.keys:
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            raise Refusal(f"{where}: a registry key is not a string literal")
        out.append(k.value)
    return out


def _default_of(name: str, value: ast.AST) -> str | None:
    """The literal default of ``os.getenv(name, <literal>)`` inside a getter."""
    for node in ast.walk(value):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        fname = ""
        if isinstance(func, ast.Attribute):
            fname = func.attr
        elif isinstance(func, ast.Name):
            fname = func.id
        if fname not in ("getenv", "get"):
            continue
        a0, a1 = node.args[0], node.args[1]
        if isinstance(a0, ast.Constant) and a0.value == name and isinstance(a1, ast.Constant):
            return None if a1.value is None else str(a1.value)
    return None


def _parse(path: str) -> ast.Module:
    try:
        with open(path, "rb") as f:
            return ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError) as exc:
        raise Refusal(f"cannot parse {path}: {exc}") from None


def read_registry(tree: str, capability: str) -> Registry:
    if capability not in CAPABILITIES:
        raise Refusal(f"unknown capability {capability!r}")
    envs = os.path.join(tree, "vllm", "envs.py")
    sm80 = os.path.join(tree, "vllm", "sm80", "envs.py")
    if not os.path.isfile(envs):
        raise Refusal(f"engine tree {tree} has no vllm/envs.py")
    up_mod = _parse(envs)
    up_node = _dict_node(up_mod, "environment_variables")
    if up_node is None:
        raise Refusal(f"{envs}: no environment_variables table")
    upstream = _keys(up_node, envs)
    defaults = {k: _default_of(k, v) for k, v in zip(upstream, up_node.values)}
    fork: list[str] = []
    retired: dict[str, str] = {}
    sm80_sha = None
    if capability == "sm80":
        if not os.path.isfile(sm80):
            raise Refusal(f"capability sm80 but engine tree {tree} has no vllm/sm80/envs.py")
        mod = _parse(sm80)
        fnode = _dict_node(mod, "FORK_ENV_VARS")
        if fnode is None:
            raise Refusal(f"{sm80}: no FORK_ENV_VARS table")
        fork = _keys(fnode, sm80)
        defaults.update({k: _default_of(k, v) for k, v in zip(fork, fnode.values)})
        rnode = _dict_node(mod, "RETIRED_ENV_VARS")
        if rnode is not None:
            for k, v in zip(_keys(rnode, sm80), rnode.values):
                try:
                    retired[k] = str(ast.literal_eval(v))
                except ValueError:
                    retired[k] = "retired"
        sm80_sha = sha256_file(sm80)
    return Registry(
        tree=tree, capability=capability, upstream=set(upstream), fork=set(fork),
        retired=retired, defaults=defaults, envs_sha256=sha256_file(envs),
        sm80_envs_sha256=sm80_sha,
    )


@dataclass
class ArgOptions:
    """Long option names (without ``--``) and alias → first name (argparse
    takes an argument's destination from its first long name)."""
    options: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)


def read_arg_options(tree: str) -> ArgOptions:
    out = ArgOptions(options={"help"})          # argparse's own
    for rel in ARG_SOURCES:
        path = os.path.join(tree, rel)
        if not os.path.isfile(path):
            continue
        module = _parse(path)
        out.options |= _generated_options(module)
        for node in ast.walk(module):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            names = [a.value[2:] for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)
                     and a.value.startswith("--")]
            out.options |= set(names)
            for alias in names[1:]:
                out.aliases[alias] = names[0]
    return out


def _generates(cls: ast.ClassDef) -> bool:
    """The class's own add_cli_args registers an option whose name is built
    at run time (``f"--{key.replace('_', '-')}"`` over ``get_kwargs(cls)``)."""
    for fn in cls.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "add_cli_args":
            return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "add_argument"
                       and any(isinstance(a, ast.JoinedStr) for a in n.args)
                       for n in ast.walk(fn))
    return False


_PSEUDO_FIELDS = ("ClassVar", "InitVar")
_KW_ONLY = "KW_ONLY"
_DATACLASS_DECORATORS = ("dataclass", "config")


def _type_name(node: ast.AST) -> str:
    node = node.value if isinstance(node, ast.Subscript) else node
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")


def _dataclass(cls: ast.ClassDef) -> bool:
    return any(_type_name(d.func if isinstance(d, ast.Call) else d) in _DATACLASS_DECORATORS
               for d in cls.decorator_list)


def _generated_options(module: ast.Module) -> set[str]:
    """Options of the frontend classes the module installs on a parser
    (``Cls.add_cli_args(parser)``) whose add_cli_args, own or inherited
    from a base class of the same module, generates option names: one
    ``--field-name`` per field ``dataclasses.fields`` gives the class, named
    as ``key.replace('_', '-')``.

    The fields follow dataclass semantics: the class's bases are walked from
    the root down and each class-level annotated name replaces an inherited
    one, so a nearer ClassVar or InitVar declaration removes an inherited
    field of the same name. A KW_ONLY sentinel annotation is no field at
    all: it neither declares a name nor removes an inherited field of the
    same name. Only shapes this reads exactly are
    accepted; anything else (a base defined elsewhere, several bases, a
    class in the chain that is not a dataclass, a string annotation) is a
    refusal, so the inventory is never wrong without saying so."""
    classes = {c.name: c for c in module.body if isinstance(c, ast.ClassDef)}

    def chain(name: str) -> list[ast.ClassDef]:
        """The class and its bases, nearest first (single inheritance only)."""
        out = []
        cls = classes[name]
        while True:
            if cls in out:
                raise Refusal(f"argument class {cls.name}: cyclic bases")
            out.append(cls)
            bases = [b for b in cls.bases if not (isinstance(b, ast.Name) and b.id == "object")]
            if not bases:
                return out
            if len(bases) != 1 or not isinstance(bases[0], ast.Name) or bases[0].id not in classes \
                    or cls.keywords:
                raise Refusal(f"argument class {cls.name}: bases "
                              f"{', '.join(ast.unparse(b) for b in cls.bases)} cannot be read "
                              "statically; its generated options are unknown")
            cls = classes[bases[0].id]

    def generating(name: str) -> bool:
        for c in chain(name):               # nearest add_cli_args decides
            if any(isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "add_cli_args"
                   for fn in c.body):
                return _generates(c)
        return False

    installed = {n.func.value.id for n in ast.walk(module)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "add_cli_args" and isinstance(n.func.value, ast.Name)
                 and n.func.value.id in classes}
    out: set[str] = set()
    for name in sorted(installed):
        if not generating(name):
            continue
        fields: dict[str, bool] = {}        # name -> a real field (not a pseudo-field)
        for c in reversed(chain(name)):
            if not _dataclass(c):
                raise Refusal(f"argument class {c.name} is not a dataclass the launcher can read; "
                              "its generated options are unknown")
            for st in c.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    if isinstance(st.annotation, ast.Constant):
                        raise Refusal(f"argument class {c.name}: string annotation of "
                                      f"{st.target.id} cannot be read statically")
                    kind = _type_name(st.annotation)
                    if kind == _KW_ONLY:
                        continue
                    fields[st.target.id] = kind not in _PSEUDO_FIELDS
        out |= {f.replace("_", "-") for f, real in fields.items() if real}
    return out


def has_sm80_plugin(tree: str) -> bool:
    """True if the tree carries the fork's sm80 package or plugin."""
    return (os.path.isdir(os.path.join(tree, "vllm", "sm80"))
            or os.path.isdir(os.path.join(tree, "plugins", "sm80")))
