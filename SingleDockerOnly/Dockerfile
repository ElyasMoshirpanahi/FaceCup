FROM python:3.8
ENV PYTHONUNBUFFERED=1
RUN mkdir /liveness
WORKDIR /liveness
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ElyasMoshirpanahi/FaceCup.git /tmp/FaceCup
RUN mv /tmp/FaceCup/* /liveness/

RUN pip --timeout=1000 install --no-cache-dir gdown
RUN gdown --id 10EKrw08j1o8pWXWGXVMnyqbsrpKrjDsz -O "/liveness/models/occlusion_detection_model.h5"

COPY requirements.txt /liveness/
RUN pip --timeout=1000 install --no-cache-dir --upgrade -r /liveness/requirements.txt

CMD python run.py
