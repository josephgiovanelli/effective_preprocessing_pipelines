docker stop effective_preprocessing_pipelines
docker rm effective_preprocessing_pipelines
docker run --name effective_preprocessing_pipelines --volume %cd%:/home --detach -t ghcr.io/josephgiovanelli/effective-preprocessing-pipelines:0.1.0
docker exec effective_preprocessing_pipelines bash ./wrapper_experiments.sh %1