# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7

RUN apt-get update && \
    apt-get install -y --no-install-recommends gfortran build-essential r-base-dev && \
    apt-get install -y wget unzip

RUN Rscript -e 'install.packages("party")'
RUN Rscript -e 'install.packages("caret")'

RUN cd home
RUN wget -c https://dataverse.harvard.edu/api/access/datafile/6544681
RUN unzip 6544681
RUN mv effective_preprocessing_pipelines effective_preprocessing_pipelines_data

RUN mkdir effective_preprocessing_pipelines
WORKDIR /home/effective_preprocessing_pipelines
COPY experiment experiment
COPY resources resources
COPY scripts scripts
COPY requirements.txt .
RUN chmod 777 scripts/*
RUN mv /home/effective_preprocessing_pipelines_data/resources/raw_results /home/effective_preprocessing_pipelines/resources/raw_results

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# Install pip requirements
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ENTRYPOINT ["./scripts/wrapper_experiments.sh"]