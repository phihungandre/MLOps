import sqlite3
from flask import Flask, request, jsonify, abort
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import mlflow
import mlflow.pytorch

app = Flask(__name__)

#openai.api_key = 'your_api_key'

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

mlflow.set_tracking_uri("http://127.0.0.1:5001")
# to track the performances write on your terminal mlflow server --host <EXTERNAL_IP> --port 5001

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

    with mlflow.start_run():
        # Log the prompt as a parameter
        mlflow.log_param("prompt", prompt)

        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        with open("generated_text.txt", "w") as f:
            f.write(generated_text)
        mlflow.log_artifact("generated_text.txt")

        mlflow.pytorch.log_model(model, "model")

    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)