(installation)=
# Installation

## Installation through PyPi
Scarf requires Python 3.14. Install with:

    uv pip install scarf[extra]

Existing Zarr v2 datasets can still be opened. New data written by Scarf uses Zarr v3 with sharding.


````{note}
On Windows you will need to run the following on PowerShell (run as Administrator):

```{code-block} powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

This will enable long path lengths on Window. Read more [here](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell)

````

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

Okay, once you have got a terminal window open, type `python --version` and press `ENTER`:

Now you may see one of the following three kinds of output:

- If your output shows you have `Python 3.14.0` or a more recent 3.14 release.
  In this case, you are good to go, and you can skip Step 3.
- If you have an earlier version than 3.14, see step 3.
- If you see an error containing words `not found` or `not recognized` then most
  likely you do not have a Python version installed on your system. Move to step 3.
- On Windows you may see a blank line which means that either Python is not installed or that Windows doesn't know where it is installed


**Step 3:**

To install the latest version of Python, we suggest using Miniconda. Download appropriate
version based on your operating system from here:
https://conda.io/miniconda.html (version Python 3.10 or above).

64-bit version will be suitable for most of the users. If you are confused about 64-bit or 32-bit see this nice
[post](https://www.techsoup.org/support/articles-and-how-tos/do-i-need-the-32bit-or-64bit)

Once you have downloaded Miniconda please follow their instructions
[here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
to install it.

Once installed, please confirm that you now have a Python version greater than 3.10.0 by
typing `python --version` in your terminal. (On Windows if you see a `Anaconda3` directory on your Start Menu then it
means that Python is installed, and you don't need to check on terminal)

**Additional steps for Windows:**

Windows does not come with essential build tools for compiling C and C++ code. We need to install these:
- Click on `Download Build Tools` from this webpage: https://visualstudio.microsoft.com/visual-cpp-build-tools
- Run the downloaded file and in the `Workloads` section check the module called: `Desktop development with C++`.
- Click on `Install` button on lower right corner.

That's it now you have the required build tools.


**Step 3.5 (Optional but highly recommended)**

We recommend that you first create an environment that you then install scarf into. 
If you have followed the steps above, and you are using conda you create an environment
by typing ``conda create --name scarf_env python=3.14`` in your terminal (Windows users must use Anaconda Prompt)

This will also install python 3.10 into that environment. One of the benefits of working with
environments is that you minimize the risk of some required package that you are using for
something different (e.g. if you have another package that requires numpy to be of an older
version) is being updated without you wanting it to. This way you can keep the installation
contained.

To activate the environment type `conda activate scarf_env`. 
To know more about conda's environment management check this out: 
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


**Step 4:**

Now, in your terminal, type this to install the latest version of Scarf:
`uv pip install scarf[extra]`

**Step 4.5 (Optional)**

Most users will use Scarf as in an interactive environment using Jupyter notebooks/lab.
You can install Jupyter lab like using command: `pip install jupyterlab` .
Thereafter, you can launch the Jupyter server by typing `jupyter lab`


**Additional steps for Windows:**
Run the following command before launching Jupyter server for the first time: `conda install -y pywin32`
