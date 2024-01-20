FROM python:3.7

# Install virtualenv and create a virtual environment
RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv $VIRTUAL_ENV -p python3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies, FFmpeg, and ffprobe
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get install -y ffmpeg ffprobe && \
    $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

# Expose port
ENV PORT 8501

# Run the application
CMD ["streamlit", "run", "app.py"]
