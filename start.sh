#!/bin/bash
export PORT=${PORT:-8000}
waitress-serve --host=0.0.0.0 --port=$PORT app:app
