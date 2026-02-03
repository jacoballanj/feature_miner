# Feature Miner – Git Commit Mining & Feature Prediction

This project is a Python FastAPI web application that mines Git repositories, analyzes commits, predicts high-level software features using a Large Language Model (LLM), and optionally pushes reviewed results to ECCO via its REST API.

## Requirements

- **Ollama** (for local LLM) https://ollama.com/
- **ECCO REST server** (only for ECCO push) https://github.com/jku-isse/ecco
- **ECCO Client UI** https://github.com/jku-isse/ecco-web-client

## Required Packages

Install dependencies using pip

pip install fastapi uvicorn gitpython requests pydantic jinja2

## Running the application 

python -m uvicorn feature_miner:app --reload
