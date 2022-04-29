docker stop effective_preprocessing_pipelines
docker rm effective_preprocessing_pipelines
docker build -t effective_preprocessing_pipelines .
docker run -t -d --name effective_preprocessing_pipelines -v ${PWD}:/home effective_preprocessing_pipelines
docker exec effective_preprocessing_pipelines bash ./wrapper_experiments.sh