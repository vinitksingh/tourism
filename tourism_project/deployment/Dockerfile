# Use a minimal base image with Python 3.9 installed for deployment stability.
FROM python:3.9-slim-buster

# Set the working directory inside the container to /app, where application files will reside.
WORKDIR /app

# Copy all files from the current host directory into the container's /app directory.
COPY . .

# Install Python dependencies specified in requirements.txt. --no-cache-dir optimizes image size.
RUN pip install --no-cache-dir -r requirements.txt

# Create a dedicated non-root user for security best practices.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change ownership of the /app directory to the new user and set it as the working directory.
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app.
# It binds to port 8501, makes it accessible externally (0.0.0.0), and disables XSRF protection (for public hosting).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
