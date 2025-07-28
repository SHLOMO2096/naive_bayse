# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000 for the API server
EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]