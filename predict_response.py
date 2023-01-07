import nltk
import json
import pickle
import numpy as np
import random

ignore_words =['?', '!',',','.', "'s", "'m"]

import tensorflow 
from data_preprocessing import get_stem_words 

model = tensorflow.keras.models.load_model()
intents = json.loads(open('./PRO-C120-Student-Boilerplate-Code-main/intents.json').read())
words = pickle.load(open('./PRO-C120-Student-Boilerplate-Code-main/words.pkl'),'rb')
classes = pickle.load(open('./PRO-C120-Student-Boilerplate-Code-main/classes.pkl'),'rb')

def preprocess_user_input(user_input):

    token1 = nltk.word_tokenize(user_input)
    token2 = get_stem_words(token1,ignore_words)
    token2 = sorted(list(set(token2)))
    bag = []
    bag_of_words = []

    for i in words:
        if i in token2 :
            bag_of_words.append(1)
        else : 
            bag_of_words.append(0)
    
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(user_input):

    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predict_class_label = np.argmax(prediction[0])

    return predict_class_label

def bot_response(user_input):

    predict_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predict_class_label]
    for intent in intents['intents']:

        if intent['tag'] == predicted_class :
            bot_response = random.choice(intent['response'])
            return bot_response
print("Hi I am Stella. How can I help you?") 
while True : 

    user_input = input("Type your message")
    print(user_input)
    response = bot_response(user_input)
    print("bot response",response)
