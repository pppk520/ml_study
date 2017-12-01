FROM python:3.5

RUN apt-get update

# Gym                                                                           
RUN apt-get install -y python-opengl
RUN apt-get install -y xvfb
RUN apt-get install -y cmake

# python environment
RUN pip install pip==9.0.1

# dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# cs231n
COPY courses/cs231n/assignment1/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# NLTK
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"

# install vim for later editing
RUN apt-get install -y vim

# other tools
RUN apt-get install -y psmisc

COPY .vimrc /root
COPY .bashrc /root

CMD ["/bin/bash"]
