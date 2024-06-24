docker build --platform=linux/amd64 -t video_summarizer_api -f Dockerfile .
docker tag video_summarizer_api gcr.io/aiops-338206/video_summarizer_api
docker push gcr.io/aiops-338206/video_summarizer_api