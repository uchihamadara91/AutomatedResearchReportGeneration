FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

RUN mkdir -p research_and_analyst
COPY research_and_analyst/__init__.py research_and_analyst/

RUN pip install --no-cache-dir --user -r requirements.txt

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY . .

RUN mkdir -p /app/generated_report /app/logs

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "research_and_analyst.api.main:app", "--host", "0.0.0.0", "--port", "8000"]