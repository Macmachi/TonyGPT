# TonyGPT

TonyGPT, a humanoid robot with advanced linguistic capabilities, can understand 11 languages (without an API key) and speak 100 languages with remarkable depth. This innovative open-source project showcases the seamless integration of computer vision, natural language processing, and robotics, resulting in a highly interactive and adaptable robotic companion. Moreover, its modular and well-documented code allows developers to easily customize and extend the platform to suit their specific requirements.

### Video Presentation of TonyGPT
[Demo Video v1.0.0](https://www.youtube.com/watch?v=Fdb63Qth9-Y)

### Blog articles : 
[gadgetify.com](https://www.gadgetify.com/tonygpt/)

## Facial Recognition

* Uses a pre-trained face detection model (res10_300x300_ssd_iter_140000_fp16.caffemodel) to detect faces in camera images.
* Tracks the detected face by adjusting the robot's head position using servo motors controlled by a PID algorithm.
* If no face is detected, the robot's head turns from right to left to search for people.

## Speech Recognition and Interaction with ChatGPT

* Listens to user voice commands using the speech_recognition library.
* Recognizes speech using Google's speech recognition API.
* Interacts with OpenAI's ~~GPT-4 turbo~~ GPT-4o API to generate responses to user questions.
* Maintains a conversation history for continuous context.
* Detects specific keywords like "stop", "hello", "squat", etc., and triggers corresponding actions.

## Speech Synthesis

* Converts text to speech using OpenAI's speech synthesis API.
* Plays the generated audio using the pygame library.
* Moves the robot randomly while speaking for a more natural interaction.

## Image Capture and Description

* Captures images using the camera when the user asks the robot to describe what it sees.
* Sends the captured image to ~~GPT-4 turbo~~ GPT-4o which includes vision capabilities to generate a textual description.
* Stores the generated description in a visual memory for later use in the conversation.

## Movements and Actions

* Executes predefined action groups (such as squatting, doing sit-ups, waving, etc.) in response to specific voice commands.
* Uses the hiwonder library to control servo motors and execute fluid movements.

## Future Ideas

* Whisper Implementation for Enhanced Language Support in Input
* Addition of a servo-motorized hand to extend the robot's physical capabilities
* Object manipulation (grasping and manipulating objects)
* Choreographed movements with music
* Improvement of facial recognition to detect specific faces

## Licence
Creative Commons Attribution-NonCommercial 4.0 International Public License
