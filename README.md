# EO Disaster Analyzer

A Python package for analyzing Earth Observation (EO) data for disaster management, augmented by Large Language Models (LLMs). The goal is to orchestrate the process of identifying potential disaster areas from news, fetching relevant satellite imagery, and generating analytical reports.

## Features

- **Efficient News Analysis**: Implements a two-stage LLM pipeline to first quickly filter relevant disaster news and then perform detailed, structured data extraction.
- **Configuration-driven**: Manage API keys and settings easily via environment variables (`.env`).
- **Data Providers**: Modular connectors for fetching news from Google News and preparing for EO data sources.
- **Text and Image Preprocessing**: Includes utilities for cleaning text data with spaCy and preparing satellite imagery (tiling, normalization) for vision models.
- **Structured LLM Outputs**: Uses Pydantic schemas to ensure reliable, structured data from LLM analysis.
- **Multiple Interfaces**: Interact with the tool via a Command-Line Interface (CLI) or a REST API (FastAPI).

---

## Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd llm-eo-disasters
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 3. Install the package in editable mode (recommended for development)

```bash
pip install -e ".[dev,notebook]"
```

This reads dependencies from `pyproject.toml` and installs optional groups.

### 4. Set up environment variables

Copy the example file and fill in your values (SentinelHub, OpenAI, etc.):

```bash
cp .env.example .env
# Then edit `.env` with your configuration
```

---

## Project Structure

The project follows a modern **PEP 621** layout using `pyproject.toml` and a `src/` directory.

```
eo_disaster_analyzer/
├── src/
│   └── eo_disaster_analyzer/
│       ├── __init__.py
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   └── orchestrator.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── sentinelhub_client.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── reasoning.py
│       ├── vision/
│       │   ├── __init__.py
│       │   └── segmentation.py
│       ├── cli.py
│       └── api.py
├── tests/
│   ├── __init__.py
│   ├── test_orchestrator.py
│   ├── test_data.py
│   ├── test_llm.py
│   └── test_vision.py
├── .env.example
├── pyproject.toml
├── README.md
└── .gitignore
```

> **Note:** The sample structure above matches the current naming (`eo_disaster_analyzer`) and a modern package layout (`src/`), not the older `setup.py` layout.

---

## Usage

### Command-Line Interface

Once installed, the CLI entry point is available:

```bash
eo-analyzer --help
```

### FastAPI Server

Start the API server:

```bash
uvicorn eo_disaster_analyzer.api:app --reload
```

Then open your browser at:

```
http://127.0.0.1:8000/docs
```

---

## License

MIT License — feel free to use, modify, and distribute.