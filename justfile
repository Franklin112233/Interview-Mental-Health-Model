python_version := `cat .python-version`

# Set python virtual environment
setup:
    @pyenv local {{python_version}}
    @brew install pipx && pipx ensurepath
    @pipx install "poetry==1.8.3" && poetry env use {{python_version}}
    @poetry install

# Format the code
format:
    @poetry run ruff format .

# Lint the code
lint:
    @poetry run ruff check . --fix

# Train the model
train: clean_mlflow
    -@poetry run python src/ml_src/main.py
    -@poetry run mlflow ui

# Clean out the cache files
clean_cache:
    -@find . -type f -name "*.py[co]" -delete
    -@find . -type d -name "__pycache__" -exec rm -rf {} \;
    -@find . -type d -name ".mypy_cache" -exec rm -rf {} \;
    -@find . -type d -name ".ruff_cache" -exec rm -rf {} \;
    -@find . -type d -name ".pytest_cache" -exec rm -rf {} \;

# Clean out the local mlflow experiments
clean_mlflow:
    -@rm -rf mlruns
    -@rm -rf mlartifacts