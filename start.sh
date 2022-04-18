docker stop effective_preprocessing_pipeline
docker rm effective_preprocessing_pipeline
docker build -t effective_preprocessing_pipeline .
docker run -it --name effective_preprocessing_pipeline -v $(pwd):/home effective_preprocessing_pipeline 