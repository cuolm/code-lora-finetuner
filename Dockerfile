# Use the official Python 3.13 slim image to match pyproject.toml requirement
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# 1.Install core build dependencies (make, gcc) and cleanup temporary package lists to reduce image size.
RUN apt-get update && apt-get install -y make gcc build-essential vim && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv' globally
RUN pip install --no-cache-dir uv

# 3. Copy dependency files first for caching
COPY pyproject.toml ./

# 4. Install dependencies from pyproject.toml using 'uv pip install'.
# '--system' installs packages into the global environment inside the container.
RUN uv pip install --system .

# 5. Copy the rest of the application code
COPY . .