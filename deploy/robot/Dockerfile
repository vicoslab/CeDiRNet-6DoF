# Modern reconstruction of the cloth-unfolding robot-experiment folder adapter.
# Inference only: no robot/camera drivers or motion commands.
ARG BASE_IMAGE=cedirnet-6dof:latest
FROM ${BASE_IMAGE}
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
COPY tools/serve_observations.py /app/tools/serve_observations.py
USER 1000:1000
ENTRYPOINT ["python3", "/app/tools/serve_observations.py"]
CMD ["--serve-folder", "/data", "--model-root", "/models", "--variant", "crop_rgb", "--device", "auto"]
