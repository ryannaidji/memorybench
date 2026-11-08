FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY memorybench ./memorybench
ENV LLM_BASE_URL=http://ollama:11434/v1 MEMORY_DB=/data/memory.db
EXPOSE 8000
CMD ["uvicorn", "memorybench.api:app", "--host", "0.0.0.0", "--port", "8000"]
