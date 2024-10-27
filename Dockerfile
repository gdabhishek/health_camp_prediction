#base image -- python run time
#ubuntu os runtime
FROM python:3.9-slim

#Creating working direcorty
WORKDIR /app

#copy all the files into app
COPY . /app

#update the exisitng packages
RUN apt-get update

#Install the required packages -- xgboost
RUN apt-get install libgomp1 libomp-dev  -y

#install all the libraries from requirement file
RUN pip install -r requirements.txt

#Making the port open outside continer
EXPOSE 8501

#command
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]