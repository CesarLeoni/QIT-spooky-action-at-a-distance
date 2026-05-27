FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Added missing destination
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Added missing destination

# Trigger the simulation upon container launch
CMD ["python", "main.py"]