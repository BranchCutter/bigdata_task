# Base image with PyTorch and CUDA support (if using GPU)
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the main script
CMD ["python", "main.py"]