# Minimal Dockerfile for orchestrator service
FROM python:3.10-slim
WORKDIR /app
COPY orchestrator.py .
RUN pip install --no-cache-dir flask kubernetes
EXPOSE 5000
CMD ["python", "orchestrator.py"]
