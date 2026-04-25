FROM python:3.10-slim

WORKDIR /app

# copy requirements first (layer caching)
COPY requirements.txt .

# install core dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy full project
COPY . .

# install package in editable mode so warehouse_env is importable
RUN pip install --no-cache-dir -e .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Use root app.py which runs uvicorn on port 7860
CMD ["python", "app.py"]
