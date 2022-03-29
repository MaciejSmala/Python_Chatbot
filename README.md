# Python_Chatbot

This is a python chatbot which topic revolves around Axolotls. It uses bag of words model as well as deep neural network in order to interpret user input and provide relevant answer.
Additonally, the chatbot uses aiml logic file (aiml-logic.xml) and speech recognition. The answers to user's questions are either taken from the web api: https://theaxolotlapi.netlify.app/index.html# or from ready made questions that can be found in the qa.json file. The chatbot can also check and store knowledge thanks to the pandas library (kb.csv).
On top of that, the chatbot can answer in multiple languages thanks to Azure translation API.
The chatbot is also capable of classifying images as either a lobster or an axolotl with either the Azure image classification service or made CNN.
The CNN was also tested and improved with the use of TensorBoard.

In order to run the chatbot simply execture the python code in the file named "axolotls_chatbotCW2.py"

the variables from the cnn model didnt fit in the github repo as they are larger than 100mb but the bot works fine without it.
