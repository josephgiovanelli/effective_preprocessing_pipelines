# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7

RUN apt-get update && \
    apt-get install -y --no-install-recommends gfortran build-essential r-base-dev && \
    apt-get install -y wget unzip

RUN Rscript -e 'install.packages("party")'
RUN Rscript -e 'install.packages("caret")'

RUN cd home && mkdir dump
WORKDIR /home/dump
COPY experiment experiment
COPY resources resources
COPY scripts scripts
COPY requirements.txt .
RUN wget -c https://dataverse.harvard.edu/api/access/datafile/6855929
RUN unzip 6855929
RUN rm -rf 6855929
RUN mv raw_results /home/dump/resources/raw_results

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# Install pip requirements
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

RUN chmod 777 scripts/*
RUN mkdir /home/autoprep
WORKDIR /home/autoprep
ENTRYPOINT ["sh /home/dump/scripts/wrapper_experiments.sh"]