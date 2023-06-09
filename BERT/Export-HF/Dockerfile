# Docker file to run a container that will run the hf_onnx_bert.py 
# in Python 3 for Huggingface (no GPU).

# Load tensorflow image for huggingface and Python 3. The
# transformers-cpu & transformers-gpu max out at tranformers v4.18.0.
# There are also huggingface/transformers-inference & 
# huggingface/transformers-training which max out a transformers
# v4.24.0.
FROM python:3.9.12
# FROM huggingface/transformers-cpu
# FROM huggingface/transformers-gpu

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /export-bert
WORKDIR /export-bert
ADD . /export-bert

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the hf_onnx_bert.py program.
CMD ["python3", "hf_onnx_bert.py"]