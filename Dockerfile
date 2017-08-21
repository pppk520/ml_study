FROM python:3.5

# Gym                                                                           
RUN apt-get install -y python-opengl
RUN apt-get install -y xvfb
RUN apt-get install -y cmake

# python environment
RUN pip install pip==9.0.1

# dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# NLTK
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"

# install vim for later editing
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y psmisc

COPY .vimrc /root

CMD ["/bin/bash"]
