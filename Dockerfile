# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7

RUN apt-get update && apt-get install -y --no-install-recommends gfortran build-essential r-base-dev
RUN Rscript -e 'install.packages("party")'
RUN Rscript -e 'install.packages("caret")'

#VOLUME effective_preprocessing_pipeline

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

WORKDIR /home
#COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["python", "-m", "pip", "install", "-r", "requirements.txt"]
#CMD ["python", "scenario_generator.py"]
#CMD ["./wrapper_experiments.sh"]
