# FoxyIntel

FoxyIntel is a commandline tool that uses OpenAI API to interact with ChatGPT. It reads text files, stores them to local vector database (ChromaDB) and feeds them to ChatGPT. Additionally user can ask questions using Google via Serpent API.

## Known problems

- FoxyIntel doesn't save database in local host. Data will be lost when Docker container shuts down. 

- ChatGPT option in Settings doesn't work.

- Since FoxyIntel reads only text files, PDF files must be converted to .txt with tools like `pdftotext` or https://pdftotext.com

## Installation

### Docker

Install Docker on your system. 

Clone repository from GitHub.

`git clone https://github/japkettu/FoxyIntel`

`cd FoxyIntel`

Rename `.env_example` to `.env` and add your OpenAI and Serpent API keys to `.env` file

Build docker image

`docker build -t foxy-intel .`

Run docker in interactive mode. This command mounts the home directory to root so it is possible to upload files from host environment to docker.

`docker run -v $HOME:/root -i -t foxy-intel`


