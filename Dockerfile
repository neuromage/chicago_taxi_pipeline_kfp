FROM tensorflow/tfx:0.13.0

RUN apt-get update -y && apt-get install -y python3
COPY tfx_component_runner.py /bin/tfx_component_runner.py

ENTRYPOINT ["python", "/bin/tfx_component_runner.py"]