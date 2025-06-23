## Start 
import pyautogui
from PIL import Image
import pandas as pd
from groq import Groq
import torch 
from dotenv import load_dotenv
import os
import re
import json
import time
import numpy as np
import speech_recognition as sr
import pyttsx3

import requests
import torch

from icon_caption import load_resnet_model, load_yolo_model,load_florence_model,load_ocr,get_parsed_icons_captions
import json

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()

    # Adjust pause threshold (default is 0.8 seconds)
    r.pause_threshold = 3.0  # wait up to 3 seconds of silence before considering speech done
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio)
        print(f"User said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        speak("Sorry, I'm having trouble connecting to the service.")
        return ""



def Capture_ScreenShot(screenshot_path="screenshot.png"):
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    return screenshot_path



def LLM(api,user_prompt,model="llama-3.3-70b-versatile",image=None,system_prompt=None,max_new_token=512):
    if image:
            messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ]
            }
           ]
    else:
        messages=[
            {
                "role": "user",
                "content": user_prompt
            },
            ]    

    completion = api.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_new_token,
        response_format={"type": "json_object"},
    )

    response = completion.choices[0].message.content
    json_match = re.search(r'\{.*\}',response,re.DOTALL)
    if json_match:
        json_string = json_match.group(0)
        response = json.loads(json_string)
        #print("Parsed JSON: ",json_data)
    else:
        print("No valid JSON found in the response")
        response = None
    return response

    
def Icon_Selection(api,task,parsed_content_list,image=None):
    system_prompt = f"You are an excellent GUI Automation Bot that work on windows 10. Respond only in json format for example : {{\"id\" : \"838\"}}. No other text , no explanation."
    user_prompt = f"Given the icons information and current screen image provided , only return id of the icon  which needs to be clicked to complete this task : {task}. Json format : {{\"id\" : \"838\"}}\nHere is the icons information in text : {parsed_content_list}"
    # model = "meta-llama/llama-4-scout-17b-16e-instruct"
    icon_id = LLM(api,user_prompt,model="llama-3.3-70b-versatile",system_prompt=system_prompt,image=image,max_new_token=32)
    print(icon_id)
    try:
        icon_id = int(icon_id["id"])
    except:
        icon_id = int(icon_id["icon_id"])

    return icon_id
    
def Click_Icon(label_coordinates,icon_id):
    image = Image.open(screenshot_path)
    w , h = image.size

    # Convert coordinates from normalized to actual coordinates
    icon_coordinates = label_coordinates[icon_id]
    x1 , y1 , x2 , y2 = icon_coordinates
    x1 , y1 , x2 , y2 = int(x1 * w) , int(y1 * h) , int(x2 * w) , int(y2 * h)


    # Click at centre of icon 
    x = (x1+x2)//2
    y = (y1+y2)//2
    # pyautogui.moveTo((x1+x2)/2,(y1+y2)/2)
    pyautogui.click(x=x,y=y,clicks=2)

def Generate_Steps(task,api):
    # task = "open google chrome choose my university profile that is .nu.edu.pk , go to my google classroom and from side bar open calendar "
    template = """ Generate a structured step-by-step GUI automation plan for a given Windows 10 task.return steps only in json format
    Rules: 
    1.Ensure steps logically follow the required task and flow and no illogical step.
    2.Each action should be precise and achievable through GUI automation (Mouse click or keyboard input).
    3.Should work for any general Windows 10 GUI task. 
    4.Only return actions that can be peformed using clicks by the bot.
    Output Format: 
    { 'steps' :[{ 'action' : 'current action' } , {'action' : 'current action' }]}
    """
    system_prompt = "You are an excellent GUI Automation Bot that work on windows 10. Respond only in json format"
    user_prompt = template + f"Here is the Task return a step by step plan according to above mentioned details : {task}"
    response = LLM(groq_api,user_prompt,model="llama-3.3-70b-versatile",system_prompt=system_prompt)
    return response




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Initialize llm
    load_dotenv()  # Load environment variables from .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    groq_api = Groq(api_key=GROQ_API_KEY,)

    # Initialize llm
    load_dotenv()  # Load environment variables from .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    groq_api = Groq(api_key=GROQ_API_KEY,)

    
    # Initialize Models
    resnet_path = "resnet50.pt"
    yolo_path = "weights/icon_detect/model.pt"
    florence_path = "weights/icon_caption_florence"

    resnet_model = load_resnet_model(resnet_path)
    yolo_model = load_yolo_model(yolo_path)
    florence_model , florence_processor = load_florence_model(florence_path)
    easyocr_reader , paddle_ocr = load_ocr()

    # Enable mutithreading 
    num_cores = os.cpu_count()
    torch.set_num_threads(num_cores)

    speak("Hello, I am your assistant. How can I help you?")
    


    while True:
        speak("What task to perform?")
        user_command = listen()
        if user_command:
                if "exit" in user_command or "stop" in user_command:
                    speak("Goodbye!")
                    break
                else:
                    # task = 'currently iam on desktop which have google chrome icon , open google chrome , google chrome will open you will see two profiles choose Muhammad Ahmad profile , go to google classroom icon is alredy in shortcut so when you will open chrome you will click on classroom icon , after that go to internet of things classroom '
                    task = user_command
                    steps = Generate_Steps(task,groq_api)
                    print(steps)
                    # encoded_image = None
                    # Go to Desktop
                    pyautogui.hotkey('win', 'd')
                    
                    count = 0
                    for step in steps['steps']:
                        print(step)
                        sub_task = step['action']
                        # Capture ScreenShot
                        screenshot_path = Capture_ScreenShot()
                        time.sleep(2)  # Wait for screenshot to be saved
                        # Parse ScreenShot
                        encoded_image, icon_coordinates , icon_descriptions = get_parsed_icons_captions(screenshot_path,florence_model,florence_processor,yolo_model,paddle_ocr,easyocr_reader,resnet_model)

                        # with open(screenshot_path, 'rb') as image_file:
                        #     files = {'screenshot': image_file}
                        #     response = requests.post(url, files=files)
                        # # print("Status Code:", response.status_code)
                        # # print("Raw Response Text:\n", response.text)  # Debug line

                        # while response.status_code != 200:
                        #     print("Error in response, retrying...")
                        #     time.sleep(1)
                        #     with open(screenshot_path, 'rb') as image_file:
                        #         files = {'screenshot': image_file}
                        #         response = requests.post(url, files=files)

                        # print("Response received successfully")
                        # try:
                        #     json_data = response.json()
                        #     print("Parsed JSON:", json_data)
                        # except requests.exceptions.JSONDecodeError:
                        #     print("Response is not valid JSON")
                       
                        # Select icon
                        icon_id = Icon_Selection(groq_api,sub_task, icon_descriptions)
                        
                        # Click icon
                        Click_Icon(icon_coordinates,icon_id)
                        time.sleep(5) # Wait for the action to complete
