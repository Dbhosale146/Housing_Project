# Housing_Project
# Machine Learning Project Setup

## Software and Account Requirements
- **GitHub Account**
- **VS Code IDE**
- **GIT CLI**
- **GIT Documentation**

## Creating Conda Environment
```sh
conda create -p venv python==3.7 -y
conda activate venv/
# OR
conda activate venv
pip install -r requirements.txt
```

## Adding Files to Git
```sh
git add .
# OR
git add <file_name>
```
**Note:** To ignore specific files or folders, list them in the `.gitignore` file.

## Checking Git Status
```sh
git status
```

## Checking All Git Versions (Commits)
```sh
git log
```

## Creating a Commit
```sh
git commit -m "message"
```

## Pushing Changes to GitHub
```sh
git push origin main
```

## Checking Remote URL
```sh
git remote -v
```

## Setting Up CI/CD Pipeline in Heroku
You need the following credentials:
```sh

```

## Building a Docker Image
```sh
docker build -t <image_name>:<tagname> .
```
**Note:** Docker image names must be lowercase.

## Listing Docker Images
```sh
docker images
```

## Running Docker Image
```sh
docker run -p 5000:5000 -e PORT=5000 f8c749e73678
```

## Checking Running Containers in Docker
```sh
docker ps
```

## Stopping a Docker Container
```sh
docker stop <container_id>
```

## Installing Python Package
```sh
python setup.py install
```

## Installing `ipykernel`
```sh
pip install ipykernel