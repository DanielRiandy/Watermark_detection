FROM python:3.11.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define the environment variable for the application directory
ENV APP_HOME /app

# Set the working directory inside the container to the application directory
WORKDIR $APP_HOME

# Copy the dependencies file to the working directory
COPY requirements.txt ./

# Install any dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the local application code to the working directory inside the container
COPY . ./

# Expose the port that the application will run on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]