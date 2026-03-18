FROM python:3.9-slim

# Install the C++ compiler needed for spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the app
CMD ["gunicorn", "app:app"]
