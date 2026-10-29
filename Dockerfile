FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY python/pyproject.toml python/README.md python/LICENSE.md /app/python/
COPY python/agent_action_guard/ /app/python/agent_action_guard/
RUN python -m pip install --no-cache-dir /app/python \
    && useradd --create-home --uid 10001 aag \
    && rm -rf /app/python

USER aag
WORKDIR /home/aag

ENTRYPOINT ["agent-action-guard"]
CMD ["--help"]
