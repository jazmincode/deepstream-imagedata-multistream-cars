
import numpy as np
import cv2

def convert_box_format(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    return x1, y1, x2, y2


def generate_random_BGR_color(seed):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Generate random values for each channel (B, G, R) between 0 and 255
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    return b, g, r


def get_index(_list, key):
    indexs = [i for i, d in enumerate(_list) if key in d]
    return indexs[0] if indexs else -1


def update_trend(id, up, down):
   
    try:
        index_up = up.index(id)
    except ValueError:
        index_up = -1

    try:
        index_down = down.index(id)
    except ValueError:
        index_down = -1

    if index_up > -1 and index_down > -1:

        if  index_up > index_down:
            down.pop(index_down)
        elif index_down > index_up:
            up.pop(index_up)

    return True


def get_elements_in_common(listA, listB):
    # Convertimos las listas a conjuntos
    setA = set(listA)
    setB = set(listB)

    # La intersección de los conjuntos nos da los elementos comunes
    intersection_set = setA.intersection(setB)

    # Convertimos el conjunto resultante a una lista si es necesario
    elements_in_common = list(intersection_set)

    return elements_in_common


def is_inside_roi(bbox_points, roi_points):
    is_inside = False
    for pt in bbox_points:
        
        result = cv2.pointPolygonTest(roi_points, pt, False)
        if result > 0:
            is_inside = True
            break
    
    return is_inside


def get_direction(track_history, obj, id):
    rect_params = obj
    x = int(rect_params.left)
    y = int(rect_params.top)
    w = int(rect_params.width)
    h = int(rect_params.height)
    index=get_index(track_history,str(id))
    track = track_history[index][str(id)]
    
    track.append((float(x), float(y)))  # x, y punto central
    if len(track) > 30:  # umbral de trackking en frames
        track.pop(0)
    array_points = np.array(track)
    x, y = array_points[:, 0], array_points[:, 1]
    angles = np.arctan2(np.diff(y), np.diff(x))
    
    if np.mean(angles) > 0:
        return 'Down'
    elif np.mean(angles) < 0:
        return 'Up'
    else:
        return 'X'