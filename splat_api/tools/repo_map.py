#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a first-party import map of this repository.

Written because the tool the request named ("graphify") does not exist on PyPI and
``pydeps`` needs Graphviz plus an import-time evaluation of every module, which
means importing torch and CUDA extensions just to draw a diagram. This walks the
source with :mod:`ast` instead: no imports are executed, nothing is installed, and
the result is deterministic.

    python -m splat_api.tools.repo_map --output splat_api/docs/REPO_MAP.md
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# First-party top-level packages. Everything else is treated as external.
FIRST_PARTY = ("data_processing", "model_eval", "model_training", "scripts", "splat_api", "tests")

# Third-party imports worth calling out in the map, because they explain why a
# module is expensive or GPU-bound.
NOTABLE_EXTERNAL = (
    "torch",
    "threedgrut",
    "diffusers",
    "transformers",
    "accelerate",
    "hydra",
    "omegaconf",
    "moge",
    "h5py",
    "fastapi",
    "pydantic",
    "numpy",
    "PIL",
    "plyfile",
)


@dataclass
class ModuleInfo:
    name: str
    path: Path
    lines: int
    internal_imports: set[str] = field(default_factory=set)
    external_imports: set[str] = field(default_factory=set)
    has_main: bool = False
    docstring: str = ""


def module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_relative(name: str | None, level: int, current: str) -> str | None:
    """Turn a ``from . import x`` into an absolute module name."""
    if level == 0:
        return name
    parts = current.split(".")
    base = parts[: max(0, len(parts) - level + 1)]
    if name:
        base = base + name.split(".")
    return ".".join(base) if base else None


def top_level(name: str) -> str:
    return name.split(".", 1)[0]


def analyze(path: Path) -> ModuleInfo | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    current = module_name(path)
    info = ModuleInfo(name=current, path=path, lines=source.count("\n") + 1)
    docstring = ast.get_docstring(tree) or ""
    info.docstring = docstring.strip().splitlines()[0] if docstring.strip() else ""
    info.has_main = '__name__ == "__main__"' in source or "__name__ == '__main__'" in source

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bucket = info.internal_imports if top_level(alias.name) in FIRST_PARTY else info.external_imports
                bucket.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = resolve_relative(node.module, node.level, current)
            if not resolved:
                continue
            bucket = info.internal_imports if top_level(resolved) in FIRST_PARTY else info.external_imports
            bucket.add(resolved)
    return info


def collect(packages: tuple[str, ...]) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for package in packages:
        root = REPO_ROOT / package
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "thirdparty" in path.parts or "__pycache__" in path.parts:
                continue
            info = analyze(path)
            if info is not None:
                modules[info.name] = info
    return modules


def package_edges(modules: dict[str, ModuleInfo]) -> dict[tuple[str, str], int]:
    """Aggregate module-level imports into package-level edges with weights."""
    edges: dict[tuple[str, str], int] = defaultdict(int)
    for info in modules.values():
        source = top_level(info.name)
        for target_module in info.internal_imports:
            target = top_level(target_module)
            if target != source:
                edges[(source, target)] += 1
    return dict(edges)


def entrypoints(modules: dict[str, ModuleInfo]) -> list[ModuleInfo]:
    return sorted(
        (info for info in modules.values() if info.has_main), key=lambda info: info.name
    )


def most_depended_on(modules: dict[str, ModuleInfo], limit: int = 15) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for info in modules.values():
        for target in info.internal_imports:
            counts[target] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [(name, count) for name, count in ranked if name in modules][:limit]


def notable_external(modules: dict[str, ModuleInfo]) -> dict[str, list[str]]:
    usage: dict[str, list[str]] = {name: [] for name in NOTABLE_EXTERNAL}
    for info in modules.values():
        tops = {top_level(name) for name in info.external_imports}
        for name in NOTABLE_EXTERNAL:
            if name in tops:
                usage[name].append(info.name)
    return {name: sorted(users) for name, users in usage.items() if users}


def render(modules: dict[str, ModuleInfo]) -> str:
    lines: list[str] = [
        "<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES."
        " All rights reserved. -->",
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
        "",
        "# Repository map",
        "",
        "Generated by `python -m splat_api.tools.repo_map`. Do not edit by hand.",
        "",
        "The map exists to answer one question before touching the pipeline: which module owns a",
        "step, and what does it drag in. Edges are static `import` statements found by AST walk,",
        "so nothing here depends on a GPU or on the packages being installed.",
        "",
    ]

    package_counts: dict[str, list[ModuleInfo]] = defaultdict(list)
    for info in modules.values():
        package_counts[top_level(info.name)].append(info)

    lines += ["## Packages", "", "| Package | Modules | Lines | Role |", "| --- | --- | --- | --- |"]
    roles = {
        "data_processing": "COLMAP prep, 3DGRUT reconstruction/render, captioning, ArtiFixer3D",
        "model_eval": "Inference entrypoint, eval datasets, metrics, checkpoint loading",
        "model_training": "Training stages, pipelines, transformer, schedulers, data loaders",
        "scripts": "Standalone helper scripts",
        "splat_api": "HTTP service wrapping the pipeline (this directory)",
        "tests": "Upstream repository tests",
    }
    for package in sorted(package_counts):
        infos = package_counts[package]
        lines.append(
            f"| `{package}` | {len(infos)} | {sum(info.lines for info in infos):,} | "
            f"{roles.get(package, '')} |"
        )
    lines.append("")

    edges = package_edges(modules)
    lines += [
        "## Package dependencies",
        "",
        "Edge labels count distinct module-level imports crossing the boundary.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for package in sorted(package_counts):
        lines.append(f'    {package}["{package}"]')
    for (source, target), weight in sorted(edges.items()):
        lines.append(f"    {source} -->|{weight}| {target}")
    lines += ["```", ""]

    lines += [
        "## Command-line entrypoints",
        "",
        "| Module | Purpose |",
        "| --- | --- |",
    ]
    for info in entrypoints(modules):
        lines.append(f"| `python -m {info.name}` | {info.docstring or '-'} |")
    lines.append("")

    lines += [
        "## Most depended-on internal modules",
        "",
        "| Module | Imported by |",
        "| --- | --- |",
    ]
    for name, count in most_depended_on(modules):
        lines.append(f"| `{name}` | {count} |")
    lines.append("")

    lines += [
        "## Heavy third-party dependencies",
        "",
        "Which modules pull in the expensive parts of the stack. A module that imports",
        "`torch` or `threedgrut` cannot be loaded cheaply, which is why the API process",
        "imports neither.",
        "",
        "| Dependency | First-party modules that import it |",
        "| --- | --- |",
    ]
    for name, users in sorted(notable_external(modules).items()):
        shown = ", ".join(f"`{user}`" for user in users[:6])
        suffix = f" (+{len(users) - 6} more)" if len(users) > 6 else ""
        lines.append(f"| `{name}` | {len(users)}: {shown}{suffix} |")
    lines.append("")

    service = sorted(info for info in modules if info.startswith("splat_api.app"))
    lines += [
        "## Service module graph",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    aliases = {name: name.replace(".", "_") for name in service}
    for name in service:
        lines.append(f'    {aliases[name]}["{name.removeprefix("splat_api.app.")}"]')
    for name in service:
        for target in sorted(modules[name].internal_imports):
            if target in aliases and target != name:
                lines.append(f"    {aliases[name]} --> {aliases[target]}")
    lines += ["```", ""]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Write markdown here instead of stdout.")
    parser.add_argument(
        "--packages", nargs="*", default=list(FIRST_PARTY), help="Top-level packages to scan."
    )
    args = parser.parse_args(argv)

    modules = collect(tuple(args.packages))
    if not modules:
        print("No modules found", flush=True)
        return 1
    markdown = render(modules)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n")
        print(f"wrote {args.output} ({len(modules)} modules)", flush=True)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
