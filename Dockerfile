# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install necessary Python packages
# Ensure that requirements.txt exists in your project directory
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the working directory inside the container
WORKDIR /app

# Copy setup.py and the 'monte-carlo' package into the container
COPY setup.py /app/setup.py
COPY monte-carlo/ /app/monte-carlo/

# Compile Cython modules
RUN python setup.py build_ext --inplace

# Create the 'plots' directory inside the container
RUN mkdir -p /app/plots

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

# Set the default command to run the simulation script
CMD ["python", "monte-carlo/simulation.py"]
