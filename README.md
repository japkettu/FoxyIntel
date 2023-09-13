# FoxyIntel

FoxyIntel is a commandline tool that uses OpenAI API to interact with ChatGPT. It reads text files, stores them to local vector database (ChromaDB) and feeds them to ChatGPT. Additionally user can ask questions using Google via Serper API.

![image](https://github.com/japkettu/FoxyIntel/assets/10699748/b7e83f48-3644-44d1-b9c8-cfed4383f96f)

![image](https://github.com/japkettu/FoxyIntel/assets/10699748/95b53945-da78-4a42-a4f9-52ca26620b38)

![image](https://github.com/japkettu/FoxyIntel/assets/10699748/b731b39b-c58b-47a4-b11d-736205b9b668)



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

Rename `.env_example` to `.env` and add your OpenAI and Serper API keys to `.env` file

Build docker image

`docker build -t foxy-intel .`

Run docker in interactive mode. This command mounts the home directory to root so it is possible to upload files from host environment to docker.

On Linux

`docker run -v $HOME:/root -i -t foxy-intel`

On Windows

`docker run -v {$HOME}:/root -i -t foxy-intel`
