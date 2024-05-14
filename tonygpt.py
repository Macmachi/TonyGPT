# Version 1.0.2
# Author: Arnaud Ricci
# Project TonyGPT : A humanoid robot that understands and speaks languages with in-depth knowledge, thanks to GPT4.
# TonyGPT features: face detection, movement tracking, conversing with users using natural language processing, and performing basic physical actions based on interaction context.
# Supported languages: for speech recognition French, German, Italian, Spanish, Dutch, Russian, Portuguese, Chinese, Japanese, Korean & over 100 for voice synthesis

#!/usr/bin/python3
# coding=utf8
import math
import threading
import sys
import cv2
import time
import numpy as np
import random
import hiwonder.PID as PID
import hiwonder.Misc as Misc
import hiwonder.Board as Board
import hiwonder.Camera as Camera
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle
import openai
import speech_recognition as sr
import pygame
import io
import base64
import requests
import configparser
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
config_path = os.path.join(script_dir, 'config.ini')

config = configparser.ConfigParser()
# Please edit the INI file with personal information (check comments in INI file)
config.read(config_path)

# KEYs from the INI file
LANGUAGE_CHOSEN = config['KEYS']['LANGUAGE_CHOSEN']
API_KEY = config['KEYS']['OPENAI_API_KEY']
openai.api_key = API_KEY
client = openai.Client(api_key=openai.api_key)

if sys.version_info.major == 2:
    print('Veuillez exécuter ce programme avec python3 !')
    sys.exit(0)

# A float value representing the confidence threshold for face detection (default is 0.6).
conf_threshold = 0.8

# String variables representing the file paths of the model files.
modelFile = "/home/pi/TonyPi/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/home/pi/TonyPi/models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# A boolean variable indicating whether the program is running.
__isRunning = False
# A boolean variable indicating whether a person is detected.
detect_people = False
# A boolean variable indicating whether the robot is speaking using GPT4 api.
gptspeaking = False
# A boolean variable indicating whether the robot is speaking.
is_robot_speaking = False
# A boolean variable indicating whether a thread is analyzing voice input.
thread_voice_analyzing = False
# A boolean variable indicating whether the robot should stop listening.
stop_listening = False
# A boolean variable indicating whether the robot is moving while talking.
is_robot_talkmoving = False
# Define exchange_memory
exchange_memory = []
# Define visual_memory  
visual_memory = []
# An integer indicating the direction of the robot's head movement. initial direction (1 = right, -1 = left)
direction = 1  

# Pre-built text variable for voice synthesis (change to your language)
if LANGUAGE_CHOSEN == 'EN':
    languagePrompt = "English"
    language_codes = 'en-US'
    launch_message = "Script launched"
    farewell_message  = "Goodbye!"
    unheard_message = "I didn't hear what you said"
    not_understood_message = "Sorry, I didn't understand your question. Please try again when my left arm lifts slightly"
    greeting_message = "Hello!"
    squat_opinion = "Squats are cool!"
    abdo_opinion = "I don't like abs, but okay"
    image_description_request = "Describe this image to me"

elif LANGUAGE_CHOSEN == 'FR':
    languagePrompt = "French"
    language_codes = 'fr-FR'
    launch_message = "Script lancé"
    farewell_message  = "Aurevoir!"
    unheard_message = "Je n'ai pas entendu ce que vous avez dit"
    not_understood_message = "Désolé, je n'ai pas compris votre question. Veuillez essayer à nouveau dès que mon bras gauche se lève légèrement"
    greeting_message = "Bonjour!"
    squat_opinion = "Les squats c'est cool!"
    abdo_opinion = "J'aime pas les abdos mais ok"
    image_description_request = "Fait moi une description de cette image"

# Adding your language keywords for better understanding
stop_words = ["stop", "stoppe", "arrete", "arrête", "stop", "halt"]
greetings_keywords = ["salut", "bonjour", "hello", "hi"]
squat_keywords = ["squat", "squats", "squatting"]
abdo_keywords = ["abdo", "abdos", "abs"]
vision_keywords = ["vois", "see", "look"]

def text_to_speech(text, voice='alloy', model='tts-1', output_format='mp3'):
    """
    This function uses the OpenAI API to convert text to speech and play the result directly.
    """
    # Set is_robot_speaking to True before playing the audio file
    global is_robot_speaking
    global gptspeaking
    global is_robot_talkmoving 
    is_robot_speaking = True
    # Generate audio file
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=output_format
    )

    # Initialize Pygame
    pygame.mixer.init()

    # Load the audio file into Pygame
    if output_format == 'mp3':
        audio_file = pygame.mixer.music.load(io.BytesIO(response.read()))
    else:
        raise ValueError("Unsupported output format")

    # Play the audio file
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

        # Condition to check if the robot should move while speaking 
        if gptspeaking == True and is_robot_talkmoving == False and is_robot_speaking == True :
            is_robot_talkmoving = True
            actions = ['talk', 'talk2']
            chosen_action = random.choice(actions)
            AGC.runActionGroup(chosen_action)
            time.sleep(0.5)  
            is_robot_talkmoving = False

    # Reset is_robot_speaking to False after finishing reading the audio file
    # and reset gptspeaking to False to indicate that the robot is not speaking with gpt4 api
    is_robot_speaking = False
    gptspeaking = False

servo_data = None

def load_config():
    """
    Load servo configuration data from YAML file.
    """    
    global servo_data
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

load_config()

x_dis = servo_data['servo2']
y_dis = 1500

# Initial position
def initMove():
    Board.setPWMServoPulse(1, y_dis, 500)
    Board.setPWMServoPulse(2, x_dis, 500)

#pid initialisation
x_pid = PID.PID(P=0.45, I=0.02, D=0.02)
y_pid = PID.PID(P=0.45, I=0.02, D=0.02)

# Reset variables
def reset():
    global x_dis, y_dis
    global detect_people
    detect_people = False
    x_dis = servo_data['servo2']
    y_dis = 1500
    x_pid.clear()
    y_pid.clear()
    initMove()

# Initializes the face tracking application.
def init():
    print("Initializing Face Tracking")
    load_config()
    reset()

# Starts the face tracking application.
def start():
    global __isRunning
    __isRunning = True
    print("Starts the face tracking")

# Stops the face tracking application.
def stop():
    global __isRunning
    __isRunning = False
    reset()
    print("Stops the face tracking")

# Function for communicating with ChatGPT
def communicate_with_chatgpt(user_prompt, conversation_history, visual_memory):
    try:
        history_messages = []
        visual_description = []
        if conversation_history:
            for user, response in conversation_history:
                history_messages.append({"role": "user", "content": user})
                history_messages.append({"role": "assistant", "content": response})
        
        if visual_memory:
            print("visual memory detected")
            visual_description.append({"role": "user", "content": "visual description of what is in front of you: " + visual_memory})
            user_message = {"role": "user", "content": f"Conversation history: {history_messages} and visual description of what is in front of you: {visual_description} and the last message the user told you: {user_prompt}."}
        else:
            user_message = {"role": "user", "content": f"Conversation history: {history_messages} and the last message the user told you: {user_prompt}."}

        system_message = {
            "role": "system", "content": f"You are a humanoid robot named Tony-GPT, responding to user questions in a concise and benevolent manner in the language they request. If no language is specified, you respond in {languagePrompt}. You should never include asterisks in your responses."}

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                system_message,
                user_message
            ],
            temperature=0.7,
            stream=True,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Create a buffer to store the chunks
        buffer = ""

        for chunk in response:
            delta_content = chunk.choices[0].delta.content 
            if delta_content:
                buffer += delta_content
                # Search for punctuation in the buffer
                for punctuation in ['.', '!', '?', ';', ':']:
                    if punctuation in buffer:
                        # Split the buffer into individual sentences
                        sentences = buffer.split(punctuation)
                        for sentence in sentences:
                            # Return each sentence separately
                            sentence.strip()
                            if sentence:  
                                # Return each sentence separately
                                yield sentence
                        buffer = ""

            # If the chunk is empty, wait for the next chunk
            else:
                continue

        # If the buffer is not empty at the end, return what's left
        if buffer:
            yield buffer

    except Exception as e:
        print(f"Error communicating with ChatGPT: {e}")
        yield not_understood_message

# This function handles voice recognition and responds to user input using ChatGPT. It listens for user speech, recognizes the input, and responds accordingly. It can also capture images and generate descriptions using OpenAI's API.        
def handle_voice_recognition_and_chatgpt():
    # Global variables used in this function
    global is_robot_speaking
    global thread_voice_analyzing
    global stop_listening
    global exchange_memory
    global visual_memory
    global gptspeaking

    with sr.Microphone() as source:     
        r = sr.Recognizer()
        r.adjust_for_ambient_noise(source)
        # Print a message to indicate that the robot is listening
        print("Listening...")
        AGC.runActionGroup('listen_start') 

        try:
            # Capture user speech
            audio = r.listen(source, phrase_time_limit=8, timeout=5)

            try:
                # Recognize the user's speech using Google's speech recognition API
                user_prompt = r.recognize_google(audio, language=language_codes)
                print(f"Vous avez dit : {user_prompt}")

                if any(word in user_prompt.lower().split() for word in stop_words):
                    print("Stop keyword detected")
                    text_to_speech(farewell_message)
                    AGC.runActionGroup('listen_stop') 
                    thread_voice_analyzing = False
                
                elif any(word in user_prompt.lower().split() for word in greetings_keywords):
                    print("Greetings keyword detected")
                    text_to_speech(greeting_message)
                    AGC.runActionGroup('wave')
                    AGC.runActionGroup('listen_stop') 
                    thread_voice_analyzing = False

                elif any(word in user_prompt.lower().split() for word in squat_keywords):
                    print("Squat keyword detected")
                    text_to_speech(squat_opinion)
                    AGC.runActionGroup('squat_down')
                    AGC.runActionGroup('squat_up')
                    AGC.runActionGroup('listen_stop') 
                    thread_voice_analyzing = False

                elif any(word in user_prompt.lower().split() for word in abdo_keywords):
                    print("Abdo keyword detected")
                    text_to_speech(abdo_opinion)
                    AGC.runActionGroup('sit_ups')  
                    AGC.runActionGroup('listen_stop')                     
                    thread_voice_analyzing = False

                elif any(word in user_prompt.lower().split() for word in vision_keywords):
                    print("Capturing image...")
                    # Capture an image using the camera
                    ret, img = my_camera.read()
                    if ret:
                        face_img = img.copy()
                        face_img_encoded = cv2.imencode('.jpg', face_img)[1]
                        face_img_base64 = base64.b64encode(face_img_encoded).decode('utf-8')
                        # Send the request to the OpenAI API
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {openai.api_key}"
                        }

                        # Create a payload for the OpenAI API request
                        payload = {
                            "model": "gpt-4o",
                            "messages": [
                                {
                                "role": "user",
                                "content": [
                                    {
                                    "type": "text",
                                    "text": image_description_request
                                    },
                                    {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{face_img_base64}"
                                    }
                                    }
                                ]
                                }
                            ],
                            "max_tokens": 300
                        }

                        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                        # Parse the response and extract the generated text description
                        response_json = response.json()
                        print(response_json)
                        text_response = response_json["choices"][0]["message"]["content"]
                        gptspeaking = True
                        text_to_speech(text_response)

                        # Store the generated description in visual_memory
                        visual_memory = text_response

                else:
                    is_robot_speaking = True
                    response_parts = []
                    for response_part in communicate_with_chatgpt(user_prompt, exchange_memory, visual_memory):
                        response_parts.append(response_part)
                        print(response_part)
                        gptspeaking = True
                        text_to_speech(response_part)

                    response_message = ' '.join(response_parts)

                    # Store the entire conversation in memory
                    exchange_memory.append((user_prompt, response_message))
                    if len(exchange_memory) > 5:
                        exchange_memory.pop(0)
                    # Reset the thread_voice_analyzing variable to False in all cases
                    AGC.runActionGroup('listen_stop') 
                    thread_voice_analyzing = False
                    is_robot_speaking = False

            except sr.UnknownValueError:
                print("Sorry, I didn't understand.")
                text_to_speech(unheard_message)
                AGC.runActionGroup('listen_stop') 
                # Reset the thread_voice_analyzing variable to False in all cases
                thread_voice_analyzing = False
                is_robot_speaking = False

        except sr.WaitTimeoutError:
            print("Timeout exceeded, stopping listening.")
            AGC.runActionGroup('listen_stop') 
            # Reset the thread_voice_analyzing variable to False in all cases
            thread_voice_analyzing = False
            is_robot_speaking = False
      
size = (320, 240)
# Main Face Tracking Function
def run(img):
    global thread_voice_analyzing
    global x_dis, y_dis, detect_people
    global direction

    img_h, img_w = img.shape[:2]

    if not __isRunning:
        return img

    blob = cv2.dnn.blobFromImage(img, 1.0, (150, 150), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    detect_people = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            detect_people = True

            x1 = int(detections[0, 0, i, 3] * img_w)
            y1 = int(detections[0, 0, i, 4] * img_h)
            x2 = int(detections[0, 0, i, 5] * img_w)
            y2 = int(detections[0, 0, i, 6] * img_h)

            centerX = (x1 + x2) // 2
            centerY = (y1 + y2) // 2

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (centerX, centerY), 5, (0, 255, 0), -1)

            use_time = 0
            x_pid.SetPoint = img_w // 2
            x_pid.update(centerX)
            dx = int(x_pid.output)
            use_time = abs(dx*0.00025)
            x_dis += dx

            x_dis = 500 if x_dis < 500 else x_dis
            x_dis = 2500 if x_dis > 2500 else x_dis

            y_pid.SetPoint = img_h // 2
            y_pid.update(centerY)
            dy = int(y_pid.output)
            use_time = round(max(use_time, abs(dy*0.00025)), 5)
            y_dis += dy

            y_dis = 1000 if y_dis < 1000 else y_dis
            y_dis = 2000 if y_dis > 2000 else y_dis

            Board.setPWMServoPulse(1, y_dis, use_time*1000)
            Board.setPWMServoPulse(2, x_dis, use_time*1000)
            time.sleep(use_time)

            # If no thread is currently running to listen to the user
            if not thread_voice_analyzing:
                thread_voice_analyzing = True
                handle_voice_recognition_and_chatgpt()
                thread_voice_analyzing = False
                reset()

            # Only process the first detected face
            break  
    
    # If no face is detected, make the head turn from right to left
    if not detect_people:
        # turning right
        if direction == 1:  
            if x_dis < 2000: 
                x_dis += 20
            else:
                # change direction to left
                direction = -1
                # reset x_dis to 2000  
                x_dis = 2000  
        else:  # turning left
            if x_dis > 800: 
                x_dis -= 20
            else:
                # change direction to right
                direction = 1 
                # reset x_dis to 800 
                x_dis = 800 
        # update servo position
        Board.setPWMServoPulse(2, x_dis, 50) 
        # wait for 50ms before updating again
        time.sleep(0.05)  

    return img

# Main function
if __name__ == '__main__':
    from CameraCalibration.CalibrationConfig import *

    # Loading parameters
    param_data = np.load(calibration_param_path + '.npz')

    # Retrieving parameters
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
    
    init()
    start()

    open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')['open_once']
    if open_once:
        my_camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
    else:
        my_camera = Camera.Camera()
        my_camera.camera_open()

    text_to_speech(launch_message)
    AGC.runActionGroup('stand')
    while True:
        ret, img = my_camera.read()
        if ret:
            frame = img.copy()
            # Distortion correction
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)  
            Frame = run(frame)
            cv2.imshow('Frame', Frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)

    my_camera.camera_close()
    cv2.destroyAllWindows()
