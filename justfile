python_version := `cat .python-version`

# Set python virtual environment
setup:
    @pyenv local {{python_version}}
    @brew install pipx && pipx ensurepath
    @pipx install "poetry==1.8.3" && poetry env use {{python_version}}
    @poetry install

# Run the data preparation for sentiment analysis
run_data_preparation:
    @poetry run src/genai_src/data_preparation.py

# Run the sentiment analysis
run_sentiment:
    @poetry run src/genai_src/sentiment_analysis.py

# Run the sentiment visualisation
run_visualization:
    @poetry run src/genai_src/visualization.py

# Run the data exploration for ml model
run_exploration:
    @poetry run src/ml_src/exploration.py

# Train the machine learning model pipeline
run_train: clean_mlflow
    -@poetry run python src/ml_src/main.py
    -@poetry run mlflow ui

# Format the code
format:
    @poetry run ruff format .

# Lint the code
lint:
    @poetry run ruff check . --fix

# Run the tests
test:
    @-poetry run pytest --cov=src tests --cov-report=term-missing --cov-report=xml:cov.xml

# Clean out the local mlflow experiments
clean_mlflow:
    -@rm -rf mlruns
    -@rm -rf mlartifacts

# Clean out the cache files
clean_cache:
    -@find . -type f -name "*.py[co]" -delete
    -@find . -type d -name "__pycache__" -exec rm -rf {} \;
    -@find . -type d -name ".mypy_cache" -exec rm -rf {} \;
    -@find . -type d -name ".ruff_cache" -exec rm -rf {} \;
    -@find . -type d -name ".pytest_cache" -exec rm -rf {} \;