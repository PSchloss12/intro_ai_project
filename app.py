# app.py
from flask import Flask, render_template, request, jsonify
from generate import respond, LSTM
from random import randint
from data import get_vocab_tokenizer
import torch

app = Flask(__name__, static_url_path='/static')

device = "cpu"

vocab, tokenizer = get_vocab_tokenizer()
vocab_size = len(vocab)

embedding_dim = 1024             # 400 in the paper
hidden_dim = 1024                # 1150 in the paper
num_layers = 2                   # 3 in the paper
dropout_rate = 0.65
tie_weights = True
lr = 1e-3
model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()

path = r"C:\Users\12625\Desktop\Code\Intro_AI\jupyter_best_model-lstm.pt"
model.load_state_dict(torch.load(path, map_location=device))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        input_text = data.get('input', '')
        temperature = data.get('temperature', '')
        max_length = data.get('maxLength', '')

        # Call generate_text function
        seed = randint(0,vocab_size)
        result = respond(input_text, max_len=max_length, temp=temperature, seed=seed, device=device, tokenizer=tokenizer, vocab=vocab, model=model)
        
        return jsonify({'generatedText': result})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 501

if __name__ == '__main__':
    app.run(debug=False)
