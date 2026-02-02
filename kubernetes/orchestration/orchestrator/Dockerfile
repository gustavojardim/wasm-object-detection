# Minimal Dockerfile for orchestrator service
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libxcb1

WORKDIR /app

COPY orchestrator.py .
COPY image.png /app/samples/image.png

RUN pip install --no-cache-dir flask kubernetes opencv-python-headless numpy

EXPOSE 5000

CMD ["python", "orchestrator.py"]
