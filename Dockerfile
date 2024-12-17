# Use the official Python 3.10 image as the base
FROM python:3.10

# Install necessary Python packages
RUN pip install --no-cache-dir numpy matplotlib tqdm cython setuptools

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Compile Cython modules
RUN python setup.py build_ext --inplace

# Create the 'plots' directory inside the container
RUN mkdir -p /app/plots

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

# Set the default command to run the simulation script
CMD ["python", "monte-carlo/simulation.py"]
