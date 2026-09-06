FROM python:3.11-slim

WORKDIR /app

# System dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install application dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Expose application port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "src.app.backend:app", "--host", "0.0.0.0", "--port", "8000"]