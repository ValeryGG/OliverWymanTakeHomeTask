# Oliver Wyman Take-Home Task

## Setup

Create a virtual environment in the project root and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the pinned packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file reflects the packages installed in the original project
environment, including the Jupyter tools needed to open the notebook. The
`calendar` module used by the plotting utility is part of Python's standard
library and does not need to be installed separately.

## Run the notebook

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
  --name oliver-wyman-take-home \
  --display-name "Python (Oliver Wyman Take-Home)"
```

Start Jupyter from the project root:

```bash
python -m jupyter lab
```

Open `project_space/Untitled.ipynb`. In the notebook's kernel picker, select
**Python (Oliver Wyman Take-Home)**. You need a virtual environment with the
requirements installed, and you should select that environment's kernel so the
notebook can import the installed packages. You do not need to use the original
`python3_11` environment; create your own environment using the commands above.

