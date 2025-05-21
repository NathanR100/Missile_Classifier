FROM python:3.10-slim

# Set a working directory
WORKDIR /missile_classifier

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY src/ src/
COPY cli/ cli/
COPY models/ models/
COPY data/ data/

# Set environment variable for data directory
ENV DATA_DIR=/app/data

# Set entrypoint (update if you want to run something else)
CMD ["python", "cli/main.py"]
