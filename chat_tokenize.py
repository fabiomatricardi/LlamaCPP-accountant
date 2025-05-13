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

##### set variables and test the 2 functions ################

LLAMA_CPP_SERVER_URL = "http://127.0.0.1:8080"
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

# APPLY THE CHAT TEMPLATE
result2 = applyTemplate(LLAMA_CPP_SERVER_URL,chat)
print(result2)

# COUNT THE TOKENS
result1 = tokenize_text(LLAMA_CPP_SERVER_URL, result2)
print(result1)




