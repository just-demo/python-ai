# python-ai

https://serper.dev/
https://smith.langchain.com/
https://huggingface.co/
https://wandb.ai/

## OpenAI

https://platform.openai.com/settings/organization/usage
https://platform.openai.com/traces

## OpenAI SDK

https://openai.github.io/openai-agents-python/
```
pip install openai-agents
```

## AutoGen
```
pip install autogen-ext
pip install autogen-agentchat
```

## CrewAI
Use Python 3.12+
Upgrade pip in venv:
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```
Install crewai:
```
pip install crewai
pip install crewai-tools
```
Create project:
```
cd examples/crewai
crewai create crew demo
```

Run:
```
cd examples/crewai/demo
crewai run
```

## Ollama

http://localhost:11434/

```
ollama serve
ollama stop

ollama ls
ollama run llama3.2

ollama pull llama3.2
ollama rm llama3.2
```

## Diffusers image generation
```
pip install diffusers
pip install torch
pip install transformers
pip install accelerate
pip install "numpy<2"
```

## Hugging Face
```
pip install transformers
pip install datasets
pip install diffusers
pip install soundfile
pip install bitsandbytes
```

## Vector database
```
pip install langchain_chroma
pip install scikit-learn
pip install plotly
pip install sentence_transformers
```