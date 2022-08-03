docker stop effective_preprocessing_pipelines
docker rm effective_preprocessing_pipelines
docker build -t effective_preprocessing_pipelines .
docker run --name effective_preprocessing_pipelines --volume e:/effective_preprocessing_pipelines:/home --detach -t effective_preprocessing_pipelines
docker exec effective_preprocessing_pipelines bash ./wrapper_experiments.sh $1