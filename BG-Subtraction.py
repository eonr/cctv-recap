#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:

import os
import numpy as np
import cv2
import time as tm
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument("VID_PATH", help="Path to the video to be summarized")
parser.add_argument("--INTERVAL_BW_DIVISIONS", help="Interval between divisions to split the moving objects - more of this => longer video => less overlapping")
args = parser.parse_args()
# In[2]:

VID_PATH = args.VID_PATH

cap  = cv2.VideoCapture(VID_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

CONTINUITY_THRESHOLD = fps #For cutting out boxes

MIN_SECONDS = 2 # (seconds) Minimum duration of a moving object
INTERVAL_BW_DIVISIONS = 10 # (seconds) For distributing moving objects over a duration to reduce overlapping.
GAP_BW_DIVISIONS = 1.5 #(seconds)

if args.INTERVAL_BW_DIVISIONS:
    INTERVAL_BW_DIVISIONS = args.INTERVAL_BW_DIVISIONS

# ## Extracting boxes using BGSubtraction

# In[3]:


"""Will give boxes for each frame and simultaneously extract background"""

fgbg = cv2.createBackgroundSubtractorKNN()

ret, frame = cap.read()
all_conts = []

avg2 = np.float32(frame) #BG-Ext

fcount = -1

print("Extracting bounding boxes and background...")

with progressbar.ProgressBar(max_value=total_frames) as bar:
    while ret:
        
        fcount += 1

        bar.update(fcount)

        #Background extraction
        try:
            cv2.accumulateWeighted(frame, avg2, 0.01)
        except:
            break
        #if ret is true than no error with cap.isOpened
        ret, frame = cap.read()
        
        if ret==True:
            #apply background substraction
            fgmask = fgbg.apply(frame)  
            
            #apply contours on foreground
            (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            contours = np.array([np.array(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) >= 500])
            all_conts.append(contours)
            for c in contours:
                
                #get bounding box from countour
                (x, y, w, h) = c
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    #         cv2.imshow('rgb', frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
background = cv2.convertScaleAbs(avg2)


# ## Object tracking 

# In[4]:


def distance(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)

def get_nearest(p1, points):
    """returns index of the point in *points* that is closest to p1"""
    return np.argmin(distance(p1, points))


# In[5]:


class box:
    def __init__(self, coords, time):
        self.coords = coords #coordinates
        self.time   = time #nth frame/time
        
class moving_obj:
    def __init__(self, starting_box):
        self.boxes = [starting_box]
    
    def add_box(self, box):
        self.boxes.append(box)
    
    def last_coords(self):
        return self.boxes[-1].coords
    
    def age(self, curr_time):
        last_time = self.boxes[-1].time
        return curr_time - last_time    


# In[100]:


"""Will associate boxes into objects"""
#old - boxes in the previous frame
#new - boxes in the current frame

print("Associating boxes into objects...")

moving_objs = []

for curr_time, new_boxes in enumerate(all_conts): #iterating over frames
    if len(new_boxes) != 0: #if not empty
        new_assocs = [None]*len(new_boxes) #all new boxes initially are not associated with any moving_objs
        obj_coords = np.array([obj.last_coords() for obj in moving_objs if obj.age(curr_time)<CONTINUITY_THRESHOLD])
        unexp_idx = -1 #index of unexpired obj in moving_objs
        for obj_idx, obj in enumerate(moving_objs):
            if obj.age(curr_time) < CONTINUITY_THRESHOLD: #checking only unexpired objects
                unexp_idx += 1
                nearest_new = get_nearest(obj.last_coords(), new_boxes) #nearest box to obj
                nearest_obj = get_nearest(new_boxes[nearest_new], obj_coords) #nearest obj to box

                if nearest_obj==unexp_idx: #both closest to each-other
                    #associate
                    new_assocs[nearest_new] = obj_idx
    
    
    for new_idx, new_coords in enumerate(new_boxes):
        new_assoc = new_assocs[new_idx]
        new_box = box(new_coords, curr_time)

        if new_assoc is not None: 
            #associate new box to moving_obj
            moving_objs[new_assoc].add_box(new_box)
        else: 
            #add a fresh, new moving_obj to moving_objs
            new_moving_obj = moving_obj(new_box)
            moving_objs.append(new_moving_obj)


# In[101]:


#Removing objects that occur for a very small duration

MIN_FRAMES = MIN_SECONDS*fps

moving_objs = [obj for obj in moving_objs if (obj.boxes[-1].time-obj.boxes[0].time)>MIN_FRAMES]


# ## Overlaying moving objects on background

# In[102]:


def cut(image, coords):
    (x, y, w, h) = coords
    return image[y:y+h,x:x+w]


# In[103]:


def overlay(frame, image, coords):
    (x, y, w, h) = coords
    frame[y:y+h,x:x+w] = cut(image, coords)


# In[104]:


def sec2HMS(seconds):
    return tm.strftime('%M:%S', tm.gmtime(seconds))

def frame2HMS(n_frame, fps):
    return sec2HMS(float(n_frame)/float(fps))


# In[105]:


max_orig_len = max(obj.boxes[-1].time for obj in moving_objs)
max_duration = max((obj.boxes[-1].time - obj.boxes[0].time) for obj in moving_objs)
#max_duration of a moving_obj. This is taken as the duration of the final summary
start_times = [obj.boxes[0].time for obj in moving_objs]


N_DIVISIONS = int(max_orig_len/(INTERVAL_BW_DIVISIONS))



final_video  = [background.copy() for _ in range(max_duration+int(N_DIVISIONS*GAP_BW_DIVISIONS)+10)] #initializing frames of final video


# In[106]:


"""Crop moving objects from main video and overlay them on the background"""
cap  = cv2.VideoCapture(VID_PATH)
# fgbg = cv2.createBackgroundSubtractorMOG2()

ret, frame = cap.read() #original video
# all_conts = []

all_texts = []

vid_time = -1

fcount = -1

print("Cropping moving objects from the main video and overlay them on the bakground....")

with progressbar.ProgressBar(max_value=total_frames) as bar:
    
    while ret:
        
        fcount += 1
        bar.update(fcount)

        vid_time += 1
        ret, frame = cap.read()
        
        if ret==True:
            
            for obj_idx, mving_obj in enumerate(moving_objs):
                if mving_obj.boxes: #non-empty
                    first_box = mving_obj.boxes[0]
                    
                    if(first_box.time == vid_time):
                        final_time = first_box.time - start_times[obj_idx] + int(int(start_times[obj_idx]/int(INTERVAL_BW_DIVISIONS*fps))*GAP_BW_DIVISIONS*fps)
                        
                        overlay(final_video[final_time-1], frame, first_box.coords)
                        (x, y, w, h) = first_box.coords
                        
                        #TODO: DESIGN
    #                     all_texts.append((final_time-1, frame2HMS(first_box.time, fps), (x, y-10))) #Above
                        all_texts.append((final_time-1, frame2HMS(first_box.time, fps), (x+int(w/2), y+int(h/2)))) #Centre
        
                        del(mving_obj.boxes[0])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()


# In[107]:


#annotating moving objects
for (t, text, org) in all_texts:
    cv2.putText(final_video[t], text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 240,3 ), 1)
    #TODO: DESIGN

# ## Final video

# In[109]:

print("Writing recap video...")

filename = os.path.basename(VID_PATH).split('.')[0]
out = cv2.VideoWriter(filename+'_summary.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, (background.shape[1],background.shape[0]))

for frame in final_video:
    #cv2.imshow('Video summary',frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#     tm.sleep(1/30) #TODO: FPS
out.release()
cv2.destroyAllWindows()
cap.release()

print("Done!!")

# In[ ]:


#TODO:
#mog vs KNN

