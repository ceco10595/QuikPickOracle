# Dockerfile

# 1) Base image with Python
FROM python:3.10-slim

# 2) System deps for llama-cpp and general build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# 3) Set working directory
WORKDIR /app

# 4) Copy everything (app code, requirements, etc.)
COPY . .

# 5) Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 6) Download the GGUF once at build time from HF hub
RUN python3 - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='QuickPick/quikpick-llama3-13b-gguf',
    filename='models/llama-3-13b-Instruct-Q4_K_M.gguf',
    cache_dir='models'
)
EOF

# 7) Expose Streamlit port
EXPOSE 8501

# 8) Run the app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]