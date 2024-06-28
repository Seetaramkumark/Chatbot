from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response("msg")
    input = msg
    return get_chat_response(input)

def get_chat_response(text):
    # Initialize chat history
    chat_history_ids = None  

    # Let's chat for 5 lines
    for step in range(5):
        # Encode the new user input, add the eos_token, and return a tensor in PyTorch
        new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        # Generate a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode the last output tokens from the bot
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return response

if __name__ == '__main__':
    app.run(debug=True)
