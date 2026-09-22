FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY python/ /app/python/
RUN python -m pip install --no-cache-dir /app/python

ENTRYPOINT ["aag-classify"]
CMD ["--help"]
