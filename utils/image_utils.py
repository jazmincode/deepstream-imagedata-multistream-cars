import os
import os.path
from os import path
import time
import cv2
import numpy as np


def crop_frame(frame, obj_meta):
    rect_params = obj_meta.rect_params
    x = int(rect_params.left)
    y = int(rect_params.top)
    w = int(rect_params.width)
    h = int(rect_params.height)

    roi_obj = frame[y:y+h, x:x+w]

    return roi_obj

def save_frame(frame, _type):

    folder = f"./frames/{_type}"
    path = create_filename(folder,'frame','jpg', _type)
  
    try:
        cv2.imwrite(path, frame)
        print("Image saved successfully.")
    except cv2.error as e:
        print("<< Error saving image:", e)
        
    except OSError as e:
        print("<< Operating system Error:", e)
        # Por ejemplo, si no hay permisos de escritura
    except Exception as e:
        print("Unexpected error:", e)
  

    return True

def create_filename(folder, filename,_format,_type=None):
    
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError as error:
        print("<<Please verify folder given, the next was found:", error)
    
    # Generate a unique filename
    filename = f"{filename}_{int(time.time())}.{_format}" if _type == None else f"{filename}_{_type}_{int(time.time())}.{_format}"

    fullpath = os.path.join(folder, filename)

    return fullpath



def generate_video():

    folder = './frames/frames_video'
    images = os.listdir(folder)
    images.sort()
    
    try:
        file = folder+'/'+images[0]
        img = cv2.imread(file)
        height, width, channels = img.shape
        if img is None:
            print("<< Error. Check the path and format.")
        
    except cv2.error as error:
        print("<< Error loading image:", error)
    except Exception as e:
        print("<< An unexpected error occurred:", error)
    
    
    folder_video = "./videos"
   

    path = create_filename(folder_video,'video','mp4')

    # Define  codec and create VideoWriter obj
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    fps = 1
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    print(f"Generating video in: {path}")

    # Write frame in video
    for image in images:
        file = folder+'/'+image
        frame = cv2.imread(file)

        # Validate dimensions
        if frame.shape[:2] != (height, width):
            print(f"Error: Inconsistent dimensions in a frame: {frame.shape[:2]}. Expected: {(height, width)}")
            out.release()
            return

        #Is in RGB?
        if channels == 3 and np.array_equal(frame[..., ::-1], frame):
            print("The frame is in RGB format. Converting to BGR...")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif channels != 3:
            print(f"Error: Invalid frame format. 3 channels were expected, but they were found {channels}.")
            out.release()
            return

        try:
            out.write(frame)
        except cv2.error as error:
            print("<< Error writing video:", error)

    # Release VideoWriter obj
    out.release()
    print("Successfully generated video.")