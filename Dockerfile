
FROM jupyter/scipy-notebook:latest

# Install required Python packages 
RUN pip install numpy matplotlib scikit-image opencv-python

# Or use requirements.txt to install packages:
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Copy code and demo data into the container
COPY Z_Scope_Processing /home/jovyan/work/Z_Scope_Processing
COPY demo.tiff /home/jovyan/work/Data/demo.tiff

# Set the working directory
WORKDIR /home/jovyan/work
