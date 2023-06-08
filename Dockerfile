FROM python:3.8
ENV PYTHONUNBUFFERED=1

RUN rm -rf /tmp/FaceCup
RUN apt-get update && apt-get install -y git 
#RUN  apt-get install libgl1-mesa-glx

RUN git clone https://github.com/ElyasMoshirpanahi/FaceCup.git /tmp/FaceCup

WORKDIR /tmp/FaceCup

RUN git pull origin master

RUN mkdir /liveness

RUN mv /tmp/FaceCup/* /liveness/

RUN pip --timeout=1000 install --no-cache-dir gdown


#Downloading pre-trained model to models folder
#VGG.pt
RUN gdown  1kHaEqYoV_V8jG-fwOeo48h68nS0CSDMl -O /liveness/models/vggface2.pt

#Model.h5
RUN gdown  1VijRF3CZhRHTd0ea8U4ZsxIkUMlZWmUX -O /liveness/models/model.h5

WORKDIR /liveness

RUN pip --timeout=1000 install --no-cache-dir --upgrade -r requirements.txt

CMD python run.py

