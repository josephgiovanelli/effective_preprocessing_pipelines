docker stop effective_preprocessing_pipelines
docker rm effective_preprocessing_pipelines
docker build -t effective_preprocessing_pipelines .
docker run -it --name effective_preprocessing_pipelines -v $(pwd):/home effective_preprocessing_pipelines