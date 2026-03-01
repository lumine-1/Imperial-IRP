# How to run this project (for absolute beginners)

> You’ll create a **Conda** virtual environment from one of the provided files:
>
> - `min_enviroment.yml` → **minimal** set of packages (🎯 try this first)
> - `full_enviroment.yml` → full, pinned environment (use only if the minimal one fails on your machine)

---

## 1) Install Conda (if you don’t have it)

- Download **Miniconda** (recommended) from the official site and install it.
- Close & reopen your terminal after installation.

Check it works:

```bash
conda --version
```

If you see a version number, you’re good.

---

## 2) Get the code and open the project folder

```bash
# Example
git clone <your-repo-url>.git
cd <your-project-folder>
```

> All commands below must be run **inside** your project folder (the one that contains `min_enviroment.yml` and `full_enviroment.yml`).

---

## 3) Create the virtual environment (minimal first)

### Option A — Windows PowerShell (recommended on Windows)

```powershell
# Create environment from the MINIMAL config (try this first)
conda env create -f .\min_enviroment.yml

# Activate it
conda activate knet
```

> If the file defines a different name than `knet`, Conda will tell you. Use that name when activating:
>
> ```powershell
> conda activate <the-name-defined-inside-yml>
> ```

### Option B — macOS / Linux (Terminal)

```bash
# Create environment from the MINIMAL config (try this first)
conda env create -f ./min_enviroment.yml

# Activate it
conda activate knet
```

---

## 4) Verify the environment

```bash
python -V
python -c "import sys; print('Python OK:', sys.version)"
python -c "import torch; print('PyTorch OK:', torch.__version__)"  # if PyTorch is included
```

If these run without errors, your environment is ready.

---

## 5) Run the project

Most users will run the unified entry point:

```bash
python main.py
```

- Open `main.py` and **uncomment** the parts you want to run (data prep / train / evaluate / visualize), then save the file and run the command again.
- If there are project-specific settings (e.g., data paths), check and edit `settings.py` first.

> Tip: for long runs, keep your terminal open or use an IDE (e.g., PyCharm) to run `main.py`.

---

## 6) If the minimal environment fails → use the full one

Sometimes different machines have different drivers/compilers. If you see errors like “package not found”, “ABI mismatch”, or build failures:

**Deactivate & remove the old env (optional):**
```bash
conda deactivate
conda env remove -n knet
```

**Create from the full config:**

- **Windows PowerShell**
  ```powershell
  conda env create -f .\full_enviroment.yml
  conda activate knet
  ```

- **macOS / Linux**
  ```bash
  conda env create -f ./full_enviroment.yml
  conda activate knet
  ```

Then run:

```bash
python main.py
```

---

## 7) Switching / removing the environment

```bash
# Switch to the project env
conda activate knet

# Switch back to base
conda activate base

# Remove the env completely (careful!)
conda env remove -n knet
```

---

## 8) Troubleshooting (quick fixes)

- **`conda: command not found`**  
  Close & reopen your terminal after installing Miniconda. On Windows, use *Anaconda Prompt* or *PowerShell* after installation.

- **UnsatisfiableError / ResolvePackageNotFound**  
  Try the `full_enviroment.yml`. If it still fails, update Conda:
  ```bash
  conda update -n base -c defaults conda
  ```
  Then try again.

- **CUDA / GPU issues**  
  If your machine doesn’t have a compatible NVIDIA GPU/driver, use **CPU** only (the env files may already be CPU-compatible). If not, remove CUDA-related lines from the YAML or install a CPU-only PyTorch build.

- **Pip packages missing**  
  Some YAMLs include a `pip:` section. Conda will install those automatically. If something is still missing:
  ```bash
  pip install <package-name>
  ```

---

### That’s it!
1. Create env from **`min_enviroment.yml`** and try running.  
2. If it doesn’t work on your machine, switch to **`full_enviroment.yml`**.  
3. Run `python main.py` and toggle stages in `main.py` as needed.
