# Deepfake Detective Web App

Deepfakes have raised significant concerns due to their ability to propagate fake news, harm reputations, and have broad societal implications. This project will revolve around conducting a comprehensive investigation, implementing, and analyzing deepfake technology, with a particular focus on developing effective detection methods.

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

<img width="1440" alt="Screen Shot 2023-04-13 at 12 57 59 AM" src="https://github.com/tyeborg/deepfake-detective/assets/96035297/d2187062-9ef7-4ca0-a3f7-4052fbbcba2a">

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
