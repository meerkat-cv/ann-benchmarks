FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -rrequirements.txt

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
