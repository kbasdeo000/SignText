# SignText
A senior Capstone project that translates American Sign Language (ASL) into the written English alphabet, built from a machine learning model.

## About
The problem we are trying to solve is that Deaf and Hard of Hearing people lack technological tools that allow them to communicate with a native interface (without text or speech).  Given that most tools provide Speech → Text, Text → Speech, and Text → Text, we want to allow our target audience to use an interface that allows them to use their most native language (ASL).

So our vision is to make technology more accessible - and we’re doing that by creating a web application that translates ASL into written text, built from a machine learning model.

We have created a web application which reads an image taken by the user and outputs the ASL letter they’re signing. Our final project is composed of a fully functional UI integrated with a machine learning model to translate ASL to text.

Online deployment: http://signtext.ue.r.appspot.com/

** Online deployment may have issues with webcam permissons. You can run the application locally as an alternative.

## Running Locally:

Open two terminal consoles.

In the first terminal, go into the front-end project folder and run `npm start` to run the development server for the front-end. You will need to install the necessary dependencies and write a package.json file. 

In the second, run `python3 signtext.py` in the back-end folder. This will run the backend server. 

## Usage:
On the translate page of the web app, the user is able to utilize the timer to automatically take a screenshot of their hand producing sign language every 3 seconds. The app will send the image to the backend, which will return a letter prediction and display it in the text box. 
