FROM python:3.10

# Install dependencies
RUN pip install numpy matplotlib tqdm cython jupyter

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Compile Cython module
RUN python setup.py build_ext --inplace

# Set default command to launch Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
