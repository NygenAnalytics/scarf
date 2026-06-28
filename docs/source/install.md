(installation)=
# Installation

## Installation through PyPi

Scarf requires Python 3.14. Install [uv](https://docs.astral.sh/uv/) if needed, then run:

    uv pip install scarf[extra]

Existing Zarr v2 datasets can still be opened. New data written by Scarf uses Zarr v3 with sharding. See {ref}`data organization <data_organization>` for storage profiles and repacking.

Optional memory tuning when opening a store: `DataStore(..., mem_budget='8G')` bounds streaming tile sizes.


````{note}
On Windows you will need to run the following on PowerShell (run as Administrator):

```{code-block} powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

This will enable long path lengths on Windows. Read more [here](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell)

````

## Utilities

Convert an older Zarr v2 store to v3 with sharding:

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

## Installing Python

To use Scarf you need Python 3.14 installed.

**Step 1:**

First, check whether you already have Python installed. To do so, you need to open a terminal
window (aka command prompt).

```{eval-rst}
.. tabs::

  .. tab:: Linux

     Pressing key combination `Ctrl+Alt+T` together works on most Linux distributions.

  .. tab:: Windows

     If you have Anaconda installed and see it in the Start Menu then you can skip this step.

     Press `Win+R` keys on your keyboard. Then, type `cmd` and press `Enter`.
    

  .. tab:: MacOS

     Press `CMD+Space` to open spotlight search, and type `terminal` and hit `RETURN`.

```

**Step 2:**

Type `python --version` and press `ENTER`:

- If your output shows `Python 3.14.0` or a more recent 3.14 release, skip Step 3.
- If you have an earlier version, see step 3.
- If you see an error containing `not found` or `not recognized`, move to step 3.

**Step 3:**

We suggest Miniconda with Python 3.14. Download from https://conda.io/miniconda.html and follow the [installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

**Step 3.5 (Optional but recommended)**

Create an environment with Python 3.14:

    conda create --name scarf_env python=3.14

Activate with `conda activate scarf_env`.

**Step 4:**

Install Scarf:

    uv pip install scarf[extra]

**Step 4.5 (Optional)**

For Jupyter Lab: `uv pip install jupyterlab`, then `jupyter lab`.

**Additional steps for Windows:**

Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools) with the "Desktop development with C++" workload. Before first Jupyter launch: `conda install -y pywin32`.
