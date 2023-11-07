# correlated_llm_errors

## Setup

### Create a python environment

Create a python environemtn with the correct packages, as specified in `requirements.txt` and python3.10.

One recommended way to do this is to first create a conda environment and activate it

```
conda create --name "correlated_errors" python=3.10
conda activate correlated_error
```

then install the packages in that conda environment using pip

```
pip install -r requirements.txt
```

### Upload correct API keys

This repo requires two API keys: [OpenAI](https://platform.openai.com/docs/quickstart?context=python) and [Huggingface](https://huggingface.co/docs/hub/security-tokens). Make sure you have them.

One you save them, save the openAI key in your home directory under `~/.openai/api.key` and your huggingface token under `~/.huggingface/api.key`. It is also recommended that if you are using a shared computing system, you restrict the permissions on these folders using programs like `chmod`.

### Run the install script

In your python environment and with your API keys in the correct location, run the install script:

```source install.sh```

this will then create a setup script. You will have to rerun this setup script everytime you start in a new environment using

```source setup.sh```
