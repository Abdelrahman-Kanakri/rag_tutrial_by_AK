# Python Project Setup and Management Guide using `uv`

This guide outlines the steps to initialize, configure, and run a Python project using `uv` for dependency management. This process is designed to be reproducible for any new project.

## 1. Prerequisites

Ensure `uv` is installed on your system.

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Initialize a New Project

1.  **Create a project directory**:

    ```bash
    mkdir my-new-project
    cd my-new-project
    ```

2.  **Initialize `uv`**:
    This generates a `pyproject.toml` file.
    ```bash
    uv init --python 3.12
    ```

## 3. Managing Dependencies

### Adding Packages

Add libraries you need (e.g., `langchain`, `fastapi`, `numpy`). This updates `pyproject.toml` and creates/updates `uv.lock`.

```bash
uv add langchain langgraph python-dotenv
```

### Removing Packages

```bash
uv remove package_name
```

## 4. Environment Variables

For projects requiring API keys or sensitive configuration:

1.  **Create a `.env` file**:

    ```bash
    touch .env
    ```

    _Note: On Windows PowerShell use `New-Item .env` or just create it manually._

2.  **Add your variables**:
    ```env
    API_KEY=your_secret_key_here
    DEBUG=true
    ```
    _Ensure `.env` is in your `.gitignore`._

## 5. Syncing and Virtual Environment

To ensure your environment matches the lockfile (useful for collaborators or after cloning):

```bash
uv sync
```

This automatically creates and updates the `.venv` directory.

## 6. Generating `requirements.txt`

If you need a standard `requirements.txt` file for compatibility (e.g., for some deployment platforms):

```bash
uv export --format requirements-txt > requirements.txt
```

## 7. Running Code

You can run scripts within the project's environment without manually activating the virtual environment:

```bash
uv run python main.py
```

Or run a module:

```bash
uv run python -m my_module
```

## 8. Summary Checklist for New Projects

- [ ] `uv init --python 3.12`
- [ ] `uv add <dependencies>`
- [ ] Create `.env` (if needed)
- [ ] `uv run python <script.py>`
