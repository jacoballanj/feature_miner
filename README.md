# Feature Miner – Git Commit Mining & Feature Prediction

This project is a Python FastAPI web application that mines Git repositories, analyzes commits, predicts high-level software features using a Large Language Model (LLM), and optionally pushes reviewed results to ECCO via its REST API.


---

## Main Features

- Clones and mines Git repositories
- Extract commit metadata, changed files, and optional code patches
- Predict high-level software features using an LLM (via Ollama)
- Support single and batch prediction modes
- Human-in-the-loop feature review (accept / edit / reject)
- Push reviewed commits and features to ECCO using ECCO REST API
- Simple web UI built with FastAPI + Jinja2

---

## 🛠 Requirements

- **Ollama** (for local LLM) https://ollama.com/
- **ECCO REST server** (only for ECCO push) https://github.com/jku-isse/ecco
- **ECCO Client UI** https://github.com/jku-isse/ecco-web-client

## Required Packages

Install dependencies using pip

pip install fastapi uvicorn gitpython requests pydantic jinja2

## Running the application 

python -m uvicorn feature_miner:app --reload



