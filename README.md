# TonyGPT

TonyGPT is a humanoid robot that understands and speaks 20 languages with in-depth knowledge. Here is a detailed description of its main features:

# Video Presentation of TonyGPT

## Facial Recognition

* Uses a pre-trained face detection model (res10_300x300_ssd_iter_140000_fp16.caffemodel) to detect faces in camera images.
* Tracks the detected face by adjusting the robot's head position using servo motors controlled by a PID algorithm.
* If no face is detected, the robot's head turns from right to left to search for people.

## Speech Recognition and Interaction with ChatGPT

* Listens to user voice commands using the speech_recognition library.
* Recognizes speech using Google's speech recognition API.
* Interacts with OpenAI's GPT-4 API to generate responses to user questions.
* Maintains a conversation history for continuous context.
* Detects specific keywords like "stop", "hello", "squat", etc., and triggers corresponding actions.

## Speech Synthesis

* Converts text to speech using OpenAI's speech synthesis API.
* Plays the generated audio using the pygame library.
* Moves the robot randomly while speaking for a more natural interaction.

## Image Capture and Description

* Captures images using the camera when the user asks the robot to describe what it sees.
* Sends the captured image to OpenAI's GPT-4 API to generate a textual description.
* Stores the generated description in a visual memory for later use in the conversation.

## Movements and Actions

* Executes predefined action groups (such as squatting, doing sit-ups, waving, etc.) in response to specific voice commands.
* Uses the hiwonder library to control servo motors and execute fluid movements.

## Configuration and Personalization

* Loads configuration parameters from YAML and INI files.
* Allows customization of parameters such as confidence thresholds, API keys, etc.

## Future Ideas

* Addition of a servo-motorized arm to extend the robot's physical capabilities.
* Improvement of facial recognition to detect specific faces.
* Integration of new features to make the robot even more versatile.

TonyGPT is an open-source project that demonstrates the integration of computer vision, natural language processing, and robotics to create an interactive and versatile robotic companion. The code is modular and well-documented, allowing developers to extend and adapt it to their specific needs.
