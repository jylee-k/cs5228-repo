# Gemini Project Instructions

This file contains foundational mandates for Gemini CLI. Instructions here take absolute precedence over general defaults.

## Project Overview
- **Goal**: Coursework and tutorials for CS5228 (Data Mining).
- **Stack**: Python 3.11+, `uv` for dependency management, Jupyter Notebooks.
- **Key Data**: Working with census and housing datasets (`.csv`).

## Engineering Standards
- **Python Style**: Adhere to PEP 8. Use type hints for all new functions.
- **Data Handling**: 
  - Always use `pandas` for CSV manipulation.
  - When cleaning data, document the steps in the code or a markdown cell.
  - Ensure all CSV outputs use `utf-8` encoding.
- **Notebooks**:
  - Keep notebooks clean. Remove unnecessary print outputs before finishing a task.
  - Use Markdown cells to explain the logic of complex data transformations.

## Workspace Mandates
- **Testing**: If a `check_*.py` script exists for a task, run it to validate results.
- **Environment**: Use the `.venv` located in the root. Prefer `uv run` for executing scripts.
- **File Naming**: Follow the existing convention (e.g., `CW{Number}-{Question}.ipynb`).

## Common Commands
- **Run script**: `uv run <script_name>.py`
- **Install dependency**: `uv add <package>`
- **Check environment**: `uv pip list`
