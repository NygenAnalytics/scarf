(installation)=
# Installation

Scarf requires Python 3.12 or newer (`requires-python >=3.12`).

## Compiler requirement

```{note}
Scarf depends on `hnswlib` for KNN graph construction, and `hnswlib` is published only as a source
distribution, so `pip` and `uv` compile it during installation. That step needs a C++ compiler:
`build-essential` on Debian and Ubuntu, the Xcode Command Line Tools (`xcode-select --install`) on
macOS. On Windows, install [Visual C++ Build Tools] with the "Desktop development with C++"
workload.
```

On Debian or Ubuntu, install the compiler before installing Scarf:

```bash
sudo apt install build-essential
```

When the alternative pip path below uses the distribution's system Python, install its development
headers and virtual-environment support before running `python -m venv`:

```bash
sudo apt install python3-dev python3-venv
```

The unversioned packages match the distribution's default `python3`. For a separately installed
system interpreter, install the matching packages, such as `python3.12-dev` and
`python3.12-venv`. A uv-managed Python already includes its development headers and
virtual-environment support.

## Recommended: uv environment

Run these commands in the directory where you will work. [uv](https://docs.astral.sh/uv/) creates
an isolated `.venv` so Scarf does not modify the system Python:

```bash
uv venv --python 3.12
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install Scarf into the activated environment:

```bash
uv pip install "scarf[extra]"
```

The `extra` optional dependency group adds plotting (`matplotlib`, `seaborn`), AnnData support,
widgets, and other helpers used by the tutorials.

Verify the install with the Python interpreter from the same environment:

```bash
python -c "import scarf; print(scarf.__version__)"
```

## Alternative: pip environment

If uv is unavailable, first confirm that `python` is version 3.12 or newer, then create and activate
a standard virtual environment:

```bash
python -m venv .venv
```

Use the macOS, Linux, or Windows activation command above, then install and verify Scarf:

```bash
python -m pip install "scarf[extra]"
python -c "import scarf; print(scarf.__version__)"
```

## Alternative: conda environment

This path installs a prebuilt `hnswlib` from conda-forge and is recommended on Windows when Visual
C++ Build Tools are unavailable:

```bash
conda create --name scarf_env --channel conda-forge python=3.12 pip "hnswlib>=0.8"
conda activate scarf_env
python -m pip install "scarf[extra]"
python -c "import scarf; print(scarf.__version__)"
```

The conda package satisfies Scarf's `hnswlib>=0.8` requirement, so pip does not compile it.

## JupyterLab

Install and launch JupyterLab from the same activated environment as Scarf. For the uv environment:

```bash
uv pip install jupyterlab
jupyter lab
```

For the pip environment:

```bash
python -m pip install jupyterlab
jupyter lab
```

For the conda environment, install JupyterLab and register an explicit kernel:

```bash
conda install --channel conda-forge jupyterlab ipykernel
python -m ipykernel install --user --name scarf_env --display-name "Python (scarf_env)"
jupyter lab
```

Select `Python (scarf_env)` when opening a notebook from another Jupyter environment.

## Windows paths

Scarf's nested Zarr stores can exceed the default Windows path limit. Enable long paths in
PowerShell run as Administrator:

````{note}
```{code-block} powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart Windows after changing the registry so existing processes do not retain the old value.
See the [Microsoft long-path documentation].
````

## Upgrade Scarf

Activate the environment that contains Scarf, then use the same installer:

```bash
uv pip install --upgrade "scarf[extra]"
```

Or, for pip and conda environments:

```bash
python -m pip install --upgrade "scarf[extra]"
```

## Install from source

Clone the repository and let uv create an editable project environment:

```bash
git clone https://github.com/NygenAnalytics/scarf.git
cd scarf
uv sync --extra extra
uv run python -c "import scarf; print(scarf.__version__)"
```

See {doc}`developers/contributing` before making package or documentation changes.

## Troubleshooting

- **No virtual environment found:** activate `.venv`, or pass it explicitly with
  `uv pip install --python .venv "scarf[extra]"`.
- **`hnswlib` fails to build:** install the compiler and, for a system Python, matching development
  headers described above. On Windows without Build Tools, use the conda path.
- **Scarf imports in a terminal but not a notebook:** compare the terminal path from
  `python -c "import sys; print(sys.executable)"` with the notebook kernel, then launch Jupyter from
  the Scarf environment or select its registered kernel.

## Next steps

After the import check succeeds, continue with the {ref}`Quick start <quickstart>`.

[Visual C++ Build Tools]: https://visualstudio.microsoft.com/visual-cpp-build-tools
[Microsoft long-path documentation]: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
