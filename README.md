# Deepfake Detective Web App

Project Outline: Deepfakes have become an increasing concern. Particularly with regards to their potential for promoting fake news, damaging reputations and broader societal impacts. This project will focus on an investigation, implementation and analysis of deepfake technology.

## Installation
1). Clone this repository by `git clone https://github.com/tyeborg/deepfake-detective.git`.

2). Navigate to the `flaskapp` folder/change the working directory by entering the following in the command line: 
```bash
cd flaskapp
```
3). Open the Docker Application and ensure that you don't have any other containers running using `docker ps`

4). Enter the following to build the Docker container:
```bash
docker-compose up --build
```
5). Visit Deepfake Detective app at: `http://localhost:3000`

## Deepfake Detective Screencast
https://github.com/tyeborg/deepfake-detective/assets/96035297/41174918-d8ed-455c-9d99-49ad106b84e4

## Dataset Utilized
* [FaceForensics++][1] - consists of 1000 original video sequences that have been manipulated with Deepfakes.

[1]: https://www.kaggle.com/datasets/sorokin/faceforensics


## Languages & Tools Utilized

<p float="left">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=js,python,flask,html,css,docker,git,vscode" />
  </a>
</p>
