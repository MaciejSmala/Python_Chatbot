#######################################################
#  Imports
#######################################################
import speech_recognition
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

read_expr = Expression.fromstring
stemmer = LancasterStemmer()
recognizer = speech_recognition.Recognizer()

#######################################################
#  Initialise Knowledgebase. 
#######################################################
kb=[]
data = pandas.read_csv('C:/Users/smala/Desktop/pythonai/kb.csv', header=None)
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
# Main loop
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")
while True:
    #get user input
    print("Do you want to use speech recognition? [y/n]")
    choice = input("y/n: ")
    if choice == ("y"):
        print("start talking to the microphone")
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
            print("Sorry, I couldn't recognize what you said")      
    else:
        print("ok, will use keyboard input")   
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
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
        print(answer)
    else:
        # I know that x is y    
        if(tag == 'knowing'):
            try:
                params = answer[1:].split('$')
                object,subject=params[1].split(' is ')
                expr=read_expr(subject + '(' + object + ')')
                answer = ResolutionProver().prove(expr.negate(),kb,verbose=True)
                if answer:
                    print("sory, this contradicts. It is not true")
                else:
                    kb.append(expr)
                    print('ok, I will remember that',object,'is',subject)
            except ValueError:
                print("sorry, I can't get knowldege from this sentence")        
        #check that x is y
        elif(tag == 'checking'):
            try:
                params = answer[1:].split('$')
                object,subject=params[1].split(' is ')
                expr=read_expr(subject + '(' + object + ')')
                answer=ResolutionProver().prove(expr, kb, verbose=True)
                if answer:
                    print('Correct.')
                elif ResolutionProver().prove(expr.negate(),kb,verbose=True):
                    print('It is not correct.') 
                else:
                    print("I don't know that")
            except ValueError:
                print("sorry, I can't check knowdlege from this sentence")        
        #using the web api about axolotls
        elif(tag == 'api'):
            good = False
            api_url = "https://axoltlapi.herokuapp.com/"
            response = requests.get(api_url)
            if response.status_code == 200:
                axolotl = response.json()
                if axolotl:
                    print(axolotl['facts'])
                    good = True
                if not good:
                    print("Sorry, I could not resolve the location you gave me.")
        #quitting the chatbot
        elif(tag == 'bye'):
            print("Bye! Thank you for a nice conversation")
            break
        #answering the Q/A pairs
        elif(tag =='meat' or 'lenght' or 'endangered' or 'location' or 'fin'):
            for i in data["qa"]:
                if i['tag'] == tag:
                    print(i['responses'])    
        
  