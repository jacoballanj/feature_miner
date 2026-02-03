# Git Commit Mining & Feature Prediction FastApi web Application

This project mines Git commits, predicts high-level features using an Ollama LLM, supports human review, and can push results to ECCO via REST.

## Run
1. Create a venv and install dependencies
2. Start Ollama locally
3. Run:
   uvicorn feature_miner:app --reload

ECCO Rest Api
https://github.com/jku-isse/ecco

ECCO web client
https://github.com/jku-isse/ecco-web-client
