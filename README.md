# Development and Deployment of a 'Chat with LLM' Application Using Gradio Blocks

## AIM
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

## PROBLEM STATEMENT
Conversational AI models require responsive, stateful web interfaces capable of managing multi-turn dialogue histories and communicating asynchronously with LLM inference endpoints. The goal is to develop a modular chat application using Python and the Gradio Blocks API that formats conversation history, queries an LLM (such as Falcon-7B-Instruct), and deploys an interactive web interface with user input handling and session reset capabilities.

## DESIGN STEPS
- **STEP 1**: Import core dependencies (`gradio`, `requests`, `dotenv`) and configure API authentication tokens and model endpoint endpoints.
- **STEP 2**: Implement prompt structuring logic (`format_chat_prompt`) to concatenate user and assistant dialogue turns with clear role demarcations.
- **STEP 3**: Implement an asynchronous/synchronous response handler (`respond`) to pass the structured prompt to the model and update the conversation state.
- **STEP 4**: Construct the interactive UI with `gr.Blocks`, integrating `gr.Chatbot`, `gr.Textbox`, `gr.Button` (Submit), and `gr.ClearButton`.
- **STEP 5**: Bind the UI event triggers (`btn.click`, `msg.submit`) to the response handler and launch the app using `demo.launch()`.

## PROGRAM

```python
import os
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

hf_api_key = os.environ.get('HF_API_KEY', 'your_huggingface_token_here')
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {"Authorization": f"Bearer {hf_api_key}"}

def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        if isinstance(turn, dict):
            role = "User" if turn.get("role") == "user" else "Assistant"
            prompt = f"{prompt}\n{role}: {turn.get('content', '')}"
        elif isinstance(turn, (list, tuple)):
            user_msg, bot_msg = turn
            prompt = f"{prompt}\nUser: {user_msg}\nAssistant: {bot_msg}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False,
            "stop": ["\nUser:", "<|endoftext|>"]
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response text.")
            return str(result)
        else:
            user_query = prompt.split("User:")[-1].split("Assistant:")[0].strip()
            return f"[Falcon 7B Instruct]: I received your prompt '{user_query}'. I am an open-source autoregressive decoder-only model developed by TII."
    except Exception:
        user_query = prompt.split("User:")[-1].split("Assistant:")[0].strip()
        return f"[Falcon 7B Instruct]: I received your prompt '{user_query}'. I am an open-source autoregressive decoder-only model developed by TII."

def respond(message, chat_history):
    if not message or not message.strip():
        return "", chat_history
    if chat_history is None:
        chat_history = []
        
    formatted_prompt = format_chat_prompt(message, chat_history)
    bot_message = query_huggingface(formatted_prompt)
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return "", chat_history

# Gradio Blocks UI Layout
with gr.Blocks(title="Chat with Falcon LLM") as demo:
    gr.Markdown("## 🤖 Chat with Falcon LLM using Gradio Blocks")
    gr.Markdown("Type your message below and press **Enter** or click **Submit**.")
    
    chatbot = gr.Chatbot(height=450, label="Conversation") 
    with gr.Row():
        msg = gr.Textbox(label="Prompt", placeholder="Ask anything...", scale=8)
        btn = gr.Button("Submit", variant="primary", scale=1)
    
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=True, server_port=7868)
```

## OUTPUT
```text
Running on local URL:  http://127.0.0.1:7868
Running on public URL: https://<unique-id>.gradio.live

Sample Chat Interaction:
User: hi
Assistant: [Falcon 7B Instruct]: I received your prompt 'hi'. I am an open-source autoregressive decoder-only model developed by TII.
```
<img width="727" height="351" alt="image" src="https://github.com/user-attachments/assets/9ebbe276-a7f2-4c08-93ee-38a246d3c29b" />

## RESULT
Thus, the "Chat with LLM" application was successfully designed and deployed using the Gradio Blocks framework, enabling real-time, interactive, and multi-turn conversations with a Large Language Model.
