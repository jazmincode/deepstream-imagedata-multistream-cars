import cv2
import numpy as np

from utils.utils import *





def draw_counts(image,counters, label_y_start=0, label_height=50, label_color=(153, 175, 79), font_color=(255, 255, 255)):
   
    general_roi, up, down, up_roi, down_roi = counters
   

    total_roi = len(general_roi)
    up_roi_count =len(up_roi)
    down_roi_count =len(down_roi)

    up_count = len(up)
    down_count = len(down)
   

    label = f"Totales:  RoI General Vehicles={str(total_roi)} (Up={up_roi_count} and Down={str(down_roi_count)})  |  Total Vehicle Up={str(up_count)}  Total Vehicle down {str(down_count)}"
    
  
    image=cv2.rectangle(image, 
                  (0, label_y_start), 
                  (image.shape[1], label_y_start + label_height), 
                  label_color, 
                  -1)

   
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Text Size
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Text position
    margin_right = 10  
    margin_top = 10  
    text_x = image.shape[1] - text_width - margin_right
    text_y = label_y_start + text_height + margin_top

    
    image=cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, thickness)
    
    return image



def draw_trajectories(image, track_history, obj_meta):

    # get values from rect_params
    rect_params = obj_meta.rect_params
    x = int(rect_params.left)
    y = int(rect_params.top)
    w = int(rect_params.width)
    h = int(rect_params.height)

    index=get_index(track_history,str(obj_meta.object_id))
    track = track_history[index][str(obj_meta.object_id)]

    track.append((float(x), float(y)))  # x, y centroid

    if len(track) > 30:  # umbral(30 frames)
        track.pop(0)
    
    #draw trajectory
    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    image=cv2.polylines(image, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    return image
    
def draw_bounding_boxes(image, obj_meta, confidence,orientation,roi):
    
    
    # Draw ROI Up and Down
    image = cv2.polylines(image, [roi], True, (0,255,0), 3)
    

    #hide draw by deepstream
    obj_meta.text_params.display_text =  ' '
    obj_meta.text_params.text_bg_clr.set(0, 0, 0, 0.0)
    obj_meta.rect_params.border_color.set(0, 0, 0, 0.0) 


    # get values from rect_params
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)

    #parse coord
    x1,y1,x2,y2=convert_box_format(left,top,width,height)
   
    
    #get color for this obj
    b,g,r = generate_random_BGR_color(obj_meta.object_id)
    color = (b, g, r)

    #draw box with opencv
    image=cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    #draw box text
    #(b, g, r)
    label_color = (0,0,0)
    text = str(obj_meta.object_id) +' : '+ str(confidence)+' | Orientation= '+str(orientation)
    fontScale = 0.4 
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    
    image=cv2.putText(image, text, (x1, y1-10), font, fontScale, (label_color), thickness,cv2.LINE_AA)

    return image
