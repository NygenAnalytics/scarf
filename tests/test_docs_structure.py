import ast
import re
import tomllib
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DOCS_SOURCE = _REPOSITORY_ROOT / "docs" / "source"
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _toc_section(contents: str, caption: str) -> str:
    match = re.search(
        rf"(?ms)^  - caption: {re.escape(caption)}\s*$"
        rf"(?P<body>.*?)(?=^  - caption: |\Z)",
        contents,
    )
    assert match is not None, caption
    return match.group("body")


def test_llms_index_links_to_local_documentation() -> None:
    contents = (_DOCS_SOURCE / "llms.txt").read_text()
    targets = _MARKDOWN_LINK.findall(contents)

    assert targets
    for target in targets:
        assert "://" not in target
        page = target.split("#", maxsplit=1)[0]
        assert page.endswith(".html")
        source = _DOCS_SOURCE / f"{page.removesuffix('.html')}.md"
        assert source.resolve().is_relative_to(_DOCS_SOURCE.resolve())
        assert source.is_file(), target


def test_documentation_navigation_uses_learning_layers() -> None:
    toctree = (_DOCS_SOURCE / "toctree.yml").read_text()
    captions = re.findall(r"(?m)^  - caption: (.+)$", toctree)

    assert captions == [
        "Get started",
        "Core workflows",
        "Biological recipes",
        "Integration and mapping",
        "Diagnostics and method choices",
        "Data, remote, and scale",
        "Artifacts and automation",
        "Advanced examples",
        "Reference",
        "Developers",
    ]
    assert re.findall(
        r"(?m)^\s+- file: (\S+)\s*$",
        _toc_section(toctree, "Core workflows"),
    ) == [
        "tutorials/scrna_seq",
        "tutorials/scatac_seq",
        "tutorials/cite_seq",
    ]

    get_started = _toc_section(toctree, "Get started")
    artifacts = _toc_section(toctree, "Artifacts and automation")
    diagnostics = _toc_section(toctree, "Diagnostics and method choices")
    advanced = _toc_section(toctree, "Advanced examples")
    assert "analysis_with_agents" not in get_started
    assert "tutorials/agent_workflow" not in get_started
    assert "analysis_with_agents" in artifacts
    assert "tutorials/agent_workflow" in artifacts
    assert "tutorials/multimodal_diagnostics" in diagnostics
    assert "tutorials/tea_seq" in advanced
    assert "tutorials/hto_demultiplexing" in advanced


def test_focused_scanpy_and_seurat_guides_preserve_migration_routes() -> None:
    toctree = (_DOCS_SOURCE / "toctree.yml").read_text()
    landing = (_DOCS_SOURCE / "scanpy_and_seurat.md").read_text()
    conf = ast.parse((_DOCS_SOURCE / "conf.py").read_text())
    redirect_assignments = [
        ast.literal_eval(node.value)
        for node in conf.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "redirects"
            for target in node.targets
        )
    ]

    assert (_DOCS_SOURCE / "scanpy.md").is_file()
    assert (_DOCS_SOURCE / "seurat.md").is_file()
    assert "(scanpy_and_seurat)=" in landing
    assert "{doc}`scanpy`" in landing
    assert "{doc}`seurat`" in landing
    assert re.search(r"(?m)^\s+- file: scanpy\s*$", toctree)
    assert re.search(r"(?m)^\s+- file: seurat\s*$", toctree)
    assert len(redirect_assignments) == 1
    assert redirect_assignments[0]["scarf_and_scanpy"] == "scanpy.html"
    assert redirect_assignments[0]["tutorials/multimodal_integration"] == (
        "cite_seq.html#multimodal-integration"
    )


def test_agent_guide_and_llms_index_are_published() -> None:
    toctree = (_DOCS_SOURCE / "toctree.yml").read_text()
    conf = ast.parse((_DOCS_SOURCE / "conf.py").read_text())
    html_extra_path = [
        ast.literal_eval(node.value)
        for node in conf.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "html_extra_path"
            for target in node.targets
        )
    ]

    assert re.search(r"(?m)^\s*- file: analysis_with_agents\s*$", toctree)
    assert html_extra_path == [["llms.txt"]]


def test_executable_agent_workflow_is_published_and_discoverable() -> None:
    toctree = (_DOCS_SOURCE / "toctree.yml").read_text()
    workflow = (_DOCS_SOURCE / "tutorials" / "agent_workflow.md").read_text()
    discovery_routes = {
        _DOCS_SOURCE / "index.md": "{doc}`tutorials/agent_workflow`",
        _DOCS_SOURCE / "analysis_with_agents.md": ("{doc}`tutorials/agent_workflow`"),
        _DOCS_SOURCE / "llms.txt": "tutorials/agent_workflow.html",
    }

    assert re.search(r"(?m)^\s*- file: tutorials/agent_workflow\s*$", toctree)
    assert "```{code-cell} ipython3" in workflow
    for source, expected_link in discovery_routes.items():
        assert expected_link in source.read_text(), source


def test_agent_guide_is_linked_from_discovery_routes() -> None:
    routes = {
        _REPOSITORY_ROOT / "README.md": (
            "https://scarf.readthedocs.io/en/latest/analysis_with_agents.html"
        ),
        _DOCS_SOURCE / "index.md": "{doc}`analysis_with_agents`",
        _DOCS_SOURCE / "reference" / "faq.md": "{doc}`../analysis_with_agents`",
    }

    for source, expected_link in routes.items():
        assert expected_link in source.read_text(), source


def test_installation_uses_one_environment_for_install_and_runtime() -> None:
    contents = (_DOCS_SOURCE / "installation.md").read_text()

    assert "uv venv --python 3.12" in contents
    assert 'python -c "import scarf; print(scarf.__version__)"' in contents
    assert "uv pip install jupyterlab\njupyter lab" in contents
    assert "uv run jupyter lab" not in contents
    assert "pywin32" not in contents
    assert contents.index("sudo apt install python3-dev python3-venv") < contents.index(
        "python -m venv .venv"
    )
    assert "## Next steps" in contents


def test_legacy_installation_url_and_package_links_are_preserved() -> None:
    conf = ast.parse((_DOCS_SOURCE / "conf.py").read_text())
    redirect_assignments = [
        ast.literal_eval(node.value)
        for node in conf.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "redirects"
            for target in node.targets
        )
    ]
    metadata = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text())

    assert len(redirect_assignments) == 1
    assert redirect_assignments[0]["install"] == "installation.html"
    assert metadata["project"]["urls"]["Installation"].endswith("/installation.html")
    assert (
        "https://scarf.readthedocs.io/en/latest/installation.html"
        in (_REPOSITORY_ROOT / "README.md").read_text()
    )


def test_trajectory_tutorials_use_source_sink_sign_convention() -> None:
    tutorials = (
        "pseudotime.md",
        "expression_dynamics.md",
        "fate_mapping.md",
    )

    for tutorial in tutorials:
        contents = (_DOCS_SOURCE / "tutorials" / tutorial).read_text()
        assert "source_sink_vector[source] = -1.0 / source.sum()" in contents
        assert "source_sink_vector[sink] = 1.0 / sink.sum()" in contents
