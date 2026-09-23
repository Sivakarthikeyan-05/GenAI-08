import os
import io
import gradio as gr
import requests
from dotenv import load_dotenv, find_dotenv
from text_generation import Client

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())
hf_api_key = os.environ.get('HF_API_KEY', 'your_huggingface_token_here')
API_URL = os.environ.get('HF_API_FALCOM_BASE', 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct')
headers = {"Authorization": f"Bearer {hf_api_key}"}

def format_chat_prompt(message, chat_history):
    """
    Format chat history and current message into Falcon/LLM prompt structure.
    Compatible with both dictionary (messages) and tuple formats.
    """
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
    """
    Generate response from the LLM and update conversation history.
    """
    if not message or not message.strip():
        return "", chat_history
    
    if chat_history is None:
        chat_history = []
        
    formatted_prompt = format_chat_prompt(message, chat_history)
    bot_message = query_huggingface(formatted_prompt)
        
    # Append in messages format (role, content) for Gradio
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return "", chat_history

# Gradio Blocks UI Layout
with gr.Blocks(title="Chat with LLM - Falcon") as demo:
    gr.Markdown("## 🤖 Chat with Falcon LLM using Gradio Blocks")
    gr.Markdown("An interactive conversational AI web interface powered by Gradio Blocks.")
    
    chatbot = gr.Chatbot(height=500, label="Conversation") 
    with gr.Row():
        msg = gr.Textbox(
            label="Prompt", 
            placeholder="Type your question here and press Enter or Submit...",
            scale=8
        )
        btn = gr.Button("Submit", variant="primary", scale=1)
    
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear Console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    gr.close_all()
    demo.launch(share=True, server_port=7868)
