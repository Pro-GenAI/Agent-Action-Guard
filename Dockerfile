FROM python:3.13-slim

WORKDIR /app

# Install minimal build tools for some packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install project and runtime dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e .

EXPOSE 8080 8081 8000 7860

CMD ["/bin/bash"]
