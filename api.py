import sqlite3
from flask import Flask, request, jsonify, abort
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#openai.api_key = 'your_api_key'

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def check_token(token):
    conn = sqlite3.connect('tokens.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tokens WHERE token=?", (token,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

@app.route('/generate', methods=['POST'])
def generate_text():
    token = request.headers.get('Authorization')
    if not token or not check_token(token):
        abort(401)
    data = request.json
    prompt = data.get('prompt')

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)