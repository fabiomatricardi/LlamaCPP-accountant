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


### The Jinja2 option
In theory all the chat-templates are jinja templates

The difference is that every model has its own set of special tokens.
Assuming you want to use a specific model (for example Qwen3) here is the way

```bash
pip install jinja2
```

Here is the function, with the chat_template taken from the official [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/tokenizer_config.json) from the Qwen3 repository

```python
def countTokens(messages):
    from jinja2 import Template
    temp2 = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    tm = Template(temp2)
    msg=tm.render(messages=messages)
    if messages is None: return 0
    numoftokens = tokenize_text(LLAMA_CPP_SERVER_URL, msg)
    return numoftokens
```

Using the function si easy:
```python
        messages = [{"role": "user", "content": textfile}]
        num_of_tokens = countTokens(messages)
```
jinja will render the template starting from the list of messages, the same way of the function `applyTemplate(server_url: str, message: list)` but** without calling the llama-server API endpoint**.

### Jinja is powerful
It is a real language....
If you don't believe it, here the structured template from the Qwen chat_template:
```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user
query.\n\nYou are provided with function signatures within <tools></tools> XML
tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and
arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\":
<function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and
not(message.content.startswith('<tool_response>') and
message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n'
}}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not
none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content =
message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' +
reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages.role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages.role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```






