import gradio as gr
import torch
import torch.nn as nn
import json
import random
from model import NeuralNet
from nltk_utils import bow, tokenize

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bow(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

def chatbot_response(user_input):
    return get_response(user_input)

# Create Gradio interface
iface = gr.Interface(fn=chatbot_response, 
                     inputs="text", 
                     outputs="text",
                     title="Chatbot",
                     description="Hello! How can I assist you today?")

if __name__ == "__main__":
    iface.launch(share=True)
