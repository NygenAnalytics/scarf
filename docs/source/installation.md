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

Verify the install:

```bash
python -c "import scarf; print(scarf.__version__)"
```

## Jupyter

```bash
uv pip install jupyterlab
jupyter lab
```

On Windows, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools)
with the "Desktop development with C++" workload before building native wheels. Before the first
Jupyter launch in a conda environment: `conda install -y pywin32`.

````{note}
On Windows, enable long paths in PowerShell (run as Administrator):

```{code-block} powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

See the [Microsoft long-path documentation](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell).
````

## Optional: conda environment

```bash
conda create --name scarf_env python=3.12
conda activate scarf_env
uv pip install "scarf[extra]"
```

## Memory tuning

When opening a store you can bound streaming tile sizes:

```python
ds = scarf.DataStore("data.zarr", mem_budget="8G")
```

Existing Zarr v2 datasets can still be opened. New data written by Scarf uses Zarr v3 with
sharding. Storage profiles and repacking are covered in {doc}`tutorials/data_organization`
and the developers guide.

## Next steps

- {ref}`Quick start <quickstart>`
- {doc}`scarf_and_scanpy`
- {doc}`tutorials/scrna_seq`
