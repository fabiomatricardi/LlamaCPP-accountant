# LlamaCPP-accountant
How to apply chat_templates and tokenize with llama.cpp

---

## Background story
Every LLM is pre-trained on a huge amount of text data, to learn the correlation among words, concepts, sentence structure etc. They learn how to generate the next most probable word.

They become useful during the dine-tuning phase. These are the so called Instruction fine-tuned model as we know them. During this stage the model is instructed with pairs (user/assistant). To mark these pairs is where the chat templates take place.

They help to define the structure of the prompt.
They help to differentiate what the user prompt is and what the AI response should be.

Some special tokens are crucial for defining the structure of prompts used to fine-tune Large Language Models (LLMs) for conversational AI. 

As we said, when training a model, the user input and the model’s response are provided as pairs. Special tokens help the model distinguish between these parts, understanding what’s the user’s instruction and what’s supposed to be the AI’s answer. This leads to more coherent multi-turn conversations and ensures the model responds appropriately.

Here’s why we need them:

Structure: They define the overall format, telling the model where each part of the conversation begins and ends.
Differentiation: They separate user input from the AI’s response, which is crucial for the model to understand the flow of the conversation.

Here’s an example of a ChatML template:
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What's the capital of France?
<|im_end|>
<|im_start|>assistant
Paris
<|im_end|>
```
You may not believe it, but this is what we are sending to llama.cpp sever every time we call the API!

### Instructions
- we run a model with the server
- we use python to send a request
- we get back the applied chat_template


### Requirements

We need to download a model GGUF and llama.cpp, and few python packages:

the latest llama.cpp up to date llama-b5342-bin-win-vulkan-x64.zip (download it in a new directory and unzip it)
a model GGUF, let’s say we try Qwen3–0.6b, a very powerful Small Language Model, from the official ggml-org repo Qwen3-0.6B-Q8_0.gguf
install these python libraries. You don’t need a virtual environment because these ones are good to have globally
```
pip install openai requests ipython rich
```

The openai library is used to access the OpenAI compatible endpoints (not the the chat_template or the tokenize ones), requests is a popular library to send POST/GET requests over the http protocol, and rich is an amazing text interface library for the terminal.

I am installing also ipython, the interactive python tool. The easiest way to test python code.

in one terminal run the llama-server with Qwen3–0.6b
```
llama-server.exe -m Qwen3-0.6B-Q8_0.gguf
```
The Qwen3 family has a dedicated prompt format with special tokens for the instructions, like this one
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

So if we apply the chat template to our conversation messages, we have to expect something similar, right?

Here is my python function for the job: I put a lot of comments, so it is easier to understand what is happening.

```python
import json
import requests
from rich.console import Console
# set the text printed to the terminal to 90 characters
console = Console(width=90)

def applyTemplate(server_url: str, message: list):
    """
    Sends text to the llama.cpp /apply-template endpoint.Args:
        server_url: The base URL of the llama.cpp server (e.g., "http://127.0.0.1:8080").
        messages: ChatML formatted list.
    Returns:
        A string with applied tokens.
    """
    endpoint = "/apply-template"
    full_url = server_url.rstrip('/') + endpoint # Ensure no double slash
    payload = {
        "messages": message
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        # Send the POST request
        response = requests.post(full_url, headers=headers, json=payload)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        # Parse the JSON response
        res1 = response.json()
        print(res1['prompt'])
        return res1['prompt']
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to or communicating with the server at {full_url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server responded with status {e.response.status_code}: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from the server.")
        print("Response text:", response.text)
        return None
```

As per documentation we point to the llama-server endpoint /apply-template. And we will get a reply in the format:
```
{"prompt":"your chat template appliet string"}
```

This is the reason why the function returns res1[‘prompt’].

Let’s set some variables for our test:
```python
LLAMA_CPP_SERVER_URL = "http://127.0.0.1:8080"
chat = [
{"role": "user", "content": "Hello, how are you?"},
{"role": "assistant", "content": "I'm doing great. How can I help you today?"},
{"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

Now we can test it
```python
result2 = applyTemplate(LLAMA_CPP_SERVER_URL,chat)
print(result2)
```

If we run this we get something amazing:
```
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing great. How can I help you today?<|im_end|>
<|im_start|>user
I'd like to show off how chat templating works!<|im_end|>
<|im_start|>assistant
```

### Use llama.cpp to count tokens

Now that we have the messages with applied chat template, we can send the text for token counts to another end-point.

We don’t need anything else, just a new function. And we are going to test it in ipython too.

```python
def tokenize_text(server_url: str, text: str, add_special: bool = False, with_pieces: bool = False):
    """
    Sends text to the llama.cpp /tokenize endpoint.
    Args:
        server_url: The base URL of the llama.cpp server (e.g., "http://127.0.0.1:8080").
        text: The string content to tokenize.
    Returns:
        An integer with the token count, or None if an error occurred.
    """
    endpoint = "/tokenize"
    full_url = server_url.rstrip('/') + endpoint # Ensure no double slash
    # Prepare the data payload as a Python dictionary
    payload = {
        "content": text,
    }
    # Set the headers (requests usually does this automatically with json=...)
    headers = {
        "Content-Type": "application/json"
    }
    try:
        # Send the POST request
        response = requests.post(full_url, headers=headers, json=payload)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        # Parse the JSON response
        res1 = response.json()
        return len(res1['tokens'])
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to or communicating with the server at {full_url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server responded with status {e.response.status_code}: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from the server.")
        print("Response text:", response.text)
        return None
```
We take response2, containing the string with the special tokens coming from the chat template, and we send it to the tokenize endpoint
```python
result1 = tokenize_text(LLAMA_CPP_SERVER_URL, result2)
print(result1)
```







