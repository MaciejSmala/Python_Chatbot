#######################################################
#  Imports
#######################################################
from tkinter import N
import speech_recognition
from IPython.display import display, Image
#import pyttsx3
import requests
import json
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy
import pandas
import aiml
#import tensorflow
import tflearn
nltk.download('punkt')
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from tensorflow.python.framework import ops
import cv2
import tensorflow as tf

read_expr = Expression.fromstring
stemmer = LancasterStemmer()
recognizer = speech_recognition.Recognizer()

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import matplotlib.pyplot as plt
from PIL import Image
import os

#######################################################
#  Initialise Knowledgebase. 
#######################################################
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
answer = ResolutionProver().prove(None,kb,verbose=True)
if answer:
    print("Kb is inconsistent")
    import sys
    sys.exit()
else:
    print("Kb is good")    
#######################################################
#  Initialise AIML agent
#######################################################
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="aiml-logic.xml")

#load the json file with Q/A pairs
with open("qa.json") as file:
    data = json.load(file)

words = []
labels = [] #for holding tags from qa
docs_x = [] #for holding patterns
docs_y = [] #for holding intents

#looping through the qa.json file and putting contents into lists
for intent in data["qa"]:
    for pattern in intent["patterns"]:
        #tokenizing words
        wrds = nltk.word_tokenize(pattern)

        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

#stemming words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
#set removes duplicates
words = sorted(list(set(words))) 
#sorting the labels "tags"
labels = sorted(labels)

training = [] #for holding the bags of words
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = [] #the bag of words
    #stemming the words that will be put into the training
    wrds = [stemmer.stem(w.lower()) for w in doc]

    #bag of words logic: put 1 if the word exists or 0 if the word does not exist
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    #copy the out_empty list
    output_row = out_empty[:]

    output_row[labels.index(docs_y[x])] = 1

    #adding the training list to the bag of words
    training.append(bag)
    output.append(output_row)

#in order for tflearn library to work we need numpy arrays instead of lists
training = numpy.array(training)
output = numpy.array(output)


ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #input data
net = tflearn.fully_connected(net, 8) #eight neuron for the layer
net = tflearn.fully_connected(net, 8) #eight neurons for the nxt layer
#output data, softmax gives us probability for each layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") 
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=False)
model.save("model.tflearn") #saving the model to a file

def bag_of_words(input, words):
    bag = [0 for _ in range(len(words))]

    #list of tokenized words and then stemming
    words_user = nltk.word_tokenize(input) 
    words_user = [stemmer.stem(word.lower()) for word in words_user]

    #bag by default has only zeros so append 1 whenever a word occurs
    for se in words_user:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


#######################################################
# Azure Translation API
#######################################################
userLang = input("What language would you like the chatbot to respond with?: ")

cog_keyTranslate = '442eb7a8d23e46cfaa5501ecf7b39ee0'
cog_endpointTranslate = 'https://translationn.cognitiveservices.azure.com/'
cog_regionTranslate = 'uksouth'

print('Ready to use cognitive services in {} using key {}'.format(cog_regionTranslate, cog_keyTranslate))
# Create a function that makes a REST request to the Text Translation service
def translate_text(cog_regionTranslate, cog_keyTranslate, text, to_lang=userLang, from_lang='en'):
    import requests, uuid, json
    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params
    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_keyTranslate,
        'Ocp-Apim-Subscription-Region':cog_regionTranslate,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    # Add the text to be translated to the body
    body = [{
        'text': text
    }]
    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]

def printTranslated(txtt):
    text_to_translate = txtt
    translation = translate_text(cog_regionTranslate, cog_keyTranslate, text_to_translate, to_lang=userLang, from_lang='en')
    #print('{} -> {}'.format(text_to_translate,translation))
    print(translation)

#######################################################
# CNN for image classification
#######################################################
CATEGORIES = ["Axolotl", "Lobster"]

def prepare(filepath):
    IMG_SIZE = 500  
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

modelCNN = tf.keras.models.load_model("AxoLobCNN.model")


#######################################################
# Main loop
#######################################################
printTranslated("Welcome to this chat bot. Please feel free to ask questions from me!")
while True:
    #get user input
    printTranslated("Do you want to use speech recognition? Please note that speech recognition only works for English")
    choice = input("y/n: ")
    if choice == ("y"):
        printTranslated("start talking to the microphone")
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic,duration=0.2)
                audio=recognizer.listen(mic)
                text = recognizer.recognize_google(audio)
                text = text.lower()
                print(f"recognized {text}")
                #getting the user input
                userInput = text
        except speech_recognition.UnknownValueError():
            printTranslated("Sorry, I couldn't recognize what you said")      
    else:
        printTranslated("ok, will use keyboard input")   
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            printTranslated("Bye!")
            break

    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'

    #activate selected response agent from the AIML file
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    #get the bag of words, feed it to the model and get the model's response
    results = model.predict([bag_of_words(userInput, words)])
    results_index = numpy.argmax(results)
    #the tag from qa chosen by the model
    tag = labels[results_index]

    #print(results,results_index)
    #print(tag)

    #give the aiml answer if possible and if not its 31, 31 or 99
    if(answer[0]!='#'):
        printTranslated(answer)
    else:
        # I know that x is y    
        if(tag == 'knowing'):
            try:
                params = answer[1:].split('$')
                object,subject=params[1].split(' is ')
                expr=read_expr(subject + '(' + object + ')')
                answer = ResolutionProver().prove(expr.negate(),kb,verbose=True)
                if answer:
                    printTranslated("sory, this contradicts. It is not true")
                else:
                    kb.append(expr)
                    print('ok, I will remember that',object,'is',subject)
            except ValueError:
                printTranslated("sorry, I can't get knowldege from this sentence")        
        #check that x is y
        elif(tag == 'checking'):
            try:
                params = answer[1:].split('$')
                object,subject=params[1].split(' is ')
                expr=read_expr(subject + '(' + object + ')')
                answer=ResolutionProver().prove(expr, kb, verbose=True)
                if answer:
                    printTranslated('Correct.')
                elif ResolutionProver().prove(expr.negate(),kb,verbose=True):
                    printTranslated('It is not correct.') 
                else:
                    printTranslated("I don't know that")
            except ValueError:
                printTranslated("sorry, I can't check knowdlege from this sentence")        
        #using the web api about axolotls
        elif(tag == 'apiFact'):
            good = False
            api_url = "https://axoltlapi.herokuapp.com/"
            response = requests.get(api_url)
            if response.status_code == 200:
                axolotl = response.json()
                if axolotl:
                    printTranslated(axolotl['facts'])
                    good = True
                if not good:
                    printTranslated("Sorry, I could not resolve the location you gave me.")
        #elif(tag == 'apiImage'):
            #good = False
            #api_url = "https://axoltlapi.herokuapp.com/"
            #response = requests.get(api_url)
            #if response.status_code == 200:
                #axolotl = response.json()
                #if axolotl:
                    #url = axolotl['url']
                    ##image = Image.open(url)
                    ##image.show()
                    #display(Image(filename=url))
                    #good = True
                #if not good:
                    #printTranslated("Sorry, I could not resolve the location you gave me.")            
#######################################################
# Azure image classification 
#######################################################                    
        elif(tag == 'img'):
            project_id = '9fd9945b-40e8-44e5-9f24-5849daa7c0f9'
            cv_key = '14491ddba20e434891fa58f4cfa6d48d'
            cv_endpoint = 'https://courseworkhehe-prediction.cognitiveservices.azure.com/'

            model_name = 'AxoLob'
            print('Ready to predict using model {} in project {}'.format(model_name, project_id))
            path = input("please paste in image path: ")
            print(path)
            image_path=os.path.join(path)
            # Create an instance of the prediction service
            credentials = ApiKeyCredentials(in_headers={"Prediction-key": cv_key})
            custom_vision_client = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)

            image_contents = open(image_path, "rb")
            classification = custom_vision_client.classify_image(project_id, model_name, image_contents.read())
            printTranslated("Azure prediction:")
            print(classification.predictions[0].tag_name)


            #CNN
            predictionCNN = modelCNN.predict([prepare(path)])
            printTranslated("CNN prediction:")
            print(CATEGORIES[int(predictionCNN[0][0])])


        #quitting the chatbot
        elif(tag == 'bye'):
            printTranslated("Bye! Thank you for a nice conversation")
            break
        #answering the Q/A pairs
        elif(tag =='meat' or 'lenght' or 'endangered' or 'location' or 'fin'):
            for i in data["qa"]:
                if i['tag'] == tag:
                    printTranslated(str(i['responses']))    
        
  