#!/bin/bash
echo "Checking model file..."
ls dl-model/eye_state_model.keras || exit 1

echo "Launching API..."
uvicorn app:app --host 0.0.0.0 --port 80