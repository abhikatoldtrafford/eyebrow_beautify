#!/bin/bash
# Start Eyebrow Beautification API

echo "========================================"
echo "Eyebrow Beautification API"
echo "========================================"
echo ""
echo "Starting server..."
echo "API Docs: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

cd "$(dirname "$0")"
python3 -m uvicorn api.api_main:app --reload --host 0.0.0.0 --port 8000
