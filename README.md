# Hugging Face Agents Course Unit 02-03

LangGraph

- https://huggingface.co/learn/agents-course
- https://github.com/huggingface/agents-course

## Overview

## Pip Environment Setup

### Create a new virtual environment
```shell
python -m venv venv
```

### Activate the environment
- On Windows:
  ```shell
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```shell
  source venv/bin/activate
  ```

### Install dependencies from a requirements file
1. Create a `requirements.txt` file with your dependencies.
2. Run:
   ```shell
   pip install -r requirements.txt
   ```

### Export the environment configuration to a file
```shell
pip freeze > requirements.txt
```
for pip packages from a conda environment
```
pip list --format=freeze > requirements.txt
```

### Create a new virtual environment and install dependencies from the `requirements.txt` file
1. Create a new virtual environment as shown above.
2. Activate the environment.
3. Run:
   ```shell
   pip install -r requirements.txt
   ```


