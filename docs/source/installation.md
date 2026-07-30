(installation)=
# Installation

## Install with uv or pip

Scarf requires Python 3.12 or newer (`requires-python >=3.12`).

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "scarf[extra]"
```

With pip:

```bash
pip install "scarf[extra]"
```

The `extra` optional dependency group adds plotting (`matplotlib`, `seaborn`, `datashader`),
AnnData support, and Jupyter helpers used by the tutorials.

```{note}
Scarf depends on `hnswlib` for KNN graph construction, and `hnswlib` is published only as a
source distribution, so `pip` and `uv` compile it during installation. That step needs a C++
compiler: `build-essential` on Debian and Ubuntu, the Xcode Command Line Tools on macOS
(`xcode-select --install`), or
[Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools) with the
"Desktop development with C++" workload on Windows. A prebuilt binary is also available from
conda-forge: run `conda install -c conda-forge hnswlib` before installing Scarf.
```

Verify the install:

```bash
python -c "import scarf; print(scarf.__version__)"
```

## Jupyter

```bash
uv pip install jupyterlab
jupyter lab
```

On Windows, before the first Jupyter launch in a conda environment: `conda install -y pywin32`.

````{note}
On Windows, enable long paths in PowerShell (run as Administrator):

```{code-block} powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

See the [Microsoft long-path documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).
````

## Optional: conda environment

```bash
conda create --name scarf_env python=3.12
conda activate scarf_env
uv pip install "scarf[extra]"
```
