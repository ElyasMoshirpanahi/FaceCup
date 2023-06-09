#!/usr/bin/env python
# coding: utf-8

#TODO_0:See if this code can be exported/merged   to   base_line/video_proccess code

#TODO_1:Change code architecture                                                []

#TODO_2:Make sure that all 6 task handel the output other wise return -1        []

#TODO_3:use majority_vote and define lists for the first 4 BINARY(0 or 1) TASKS []

#TODO_4:Turn code into class if Possible                                        []



import cv2 
import mediapipe as mp
import numpy as np
import time
import os
import sys
import random
import warnings
from time import sleep

#TF Imports

from PIL import Image
# Loading Tensorflow Functions
# Function for loading saved model <filename.model>
from tensorflow.keras.models import load_model
# Function for transforming PIL Image to Numpy Array
from tensorflow.keras.preprocessing.image import img_to_array
# Preprocessing Function for Input Image before passing into MobileNet Model (Mask Detector Model)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')




#============================= Define Once =============================#

#---------------------------- Constatns ----------------------------#
#Number of GPUS for Pytorch
device_id = 0
occ_model_path = "./models/occlusion_detection_model.h5" # should be changed and replaced
#-------------------------------------------------------------------#


#-------------------- Face Mesh/Detection setup --------------------#
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence =0.5 ,min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence =0.4) 


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 118, 0), thickness=1,circle_radius =1)

eye_landmarks = [1, 4, 5, 246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 
                154, 155, 133, 247, 30, 29, 27, 28, 56, 190, 130, 25, 110, 24, 
                23, 22, 26, 112, 243, 113, 225]

mouth_landmarks = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146,
                    91, 181, 84, 17, 314, 405]

difference_threshold = 0.099


#-------------------------------------------------------------------#


#------------------------- Spoof model load ------------------------#
spoof_model = AntiSpoofPredict(device_id)
image_cropper = CropImage()
#occlusion_detection_model  = load_model(occ_model_path)
#-------------------------------------------------------------------#

#======================================================================#









#============================== Functions ==============================#
# Function to preprocess the frame
def preprocess_frame(frame, image_cropper, image_bbox, h_input, w_input, scale):
    param = {
        "org_img": frame,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)
    return img

# get Euclidian distance
def get_distance(point1, point2):
    x1, y1, _ = point1
    x2, y2, _ = point2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


#Important for first 4 TASKS
def majority_vote(occluded, no_occluded):
    '''
    Takes two lists and votes for the higher list sum returns an int either 1 or 0
    Important for the first 4 TASKS
    '''
    occluded_count = sum(occluded)
    no_occluded_count = sum(no_occluded)
    
    if occluded_count > no_occluded_count:
        return 1  # Face is occluded
    else:
        return 0  # Face is not occluded




# Main function , Currently supporting => spoof detection , pose estimation , mouth movements , face count , same person or not,occlusion
def main(source=0,tfps = 3, mp_face_mesh=mp_face_mesh, 
         face_mesh=face_mesh, 
         mp_face_detection=mp_face_detection,
         face_detection=face_detection,
         mp_drawing=mp_drawing,
         drawing_spec=drawing_spec,
         model_dir="./models/anti_spoof_models",
         spoof_model = spoof_model, 
         device_id=0,
         spoof_frame_threshold=15):
    """
    Runs the main loop for face detection, pose estimation, and spoof detection.

    Parameters:
        source (int or str): Source for the video stream. Defaults to 0 (the default camera).
        tfps (int): Time delay between frames in milliseconds. Defaults to 3.
        mp_face_mesh: MediaPipe face mesh model.
        face_mesh: MediaPipe face mesh instance.
        mp_face_detection: MediaPipe face detection model.
        face_detection: MediaPipe face detection instance.
        mp_drawing: MediaPipe drawing utility.
        drawing_spec: MediaPipe drawing specification.
        model_dir (str): Directory path for anti-spoof models. Defaults to "./models/anti_spoof_models".
        spoof_model: Spoof detection model.
        device_id (int): ID of the device to be used. Defaults to 0.
        spoof_frame_threshold (int): Number of frames for spoof detection. Defaults to 15.

    Returns:
        tuple: A tuple containing the detected movements list and spoof dictionary and multi_faces.
    """




    cap = cv2.VideoCapture(source)
   

   #------------------------ Variables ----------------------#


    index     = 0
    counter   = 0
    times_OCC_checked   = 0


    # Define color options for more than 1  face 
    detected_movements  = []
    fake                = []
    real                = []
    multi_faces         = []
    no_occlusion        = []
    occlusion           = []

    colors = [(0, 255, 0), (255, 192, 203), (0, 0, 255), (255, 165, 0)]
    
    spoof_dict = {}

    simple_font   = cv2.FONT_HERSHEY_SIMPLEX
    complex_font  = cv2.FONT_HERSHEY_COMPLEX



    prev_movement  = None
    label          = None
    allow_draw     = None
    face_landmarks = None
    spoof_final    = None
    new_movement  = None
    check_spoof      = False
    check_occlussion = True
    skip_video       = False #Skip video if spoof! Needs Implementation


    movement_counter = 0
    movement_index = 4
    final_result = np.zeros(19)
    previous_distances = np.array([])




    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    check_OCC_threshold = int(fps/(3 * 100))
   #---------------------------------------------------------#

   #-----------------Main Loop--------------------#
    while True:
        success , image = cap.read()
        
        if not success or skip_video == True:
            break

        start = time.time()

     #------------------ Spoof ------------------#
        #Spoof detection for only 10 frames!
        ##print(index)
        if check_spoof:
            if  index < spoof_frame_threshold:
                # Preprocess the image
                image_bbox = spoof_model.get_bbox(image)
                prediction = np.zeros((1, 3))
                test_speed = 0


                # Perform prediction using each model
                for model_name in os.listdir(model_dir):
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    img = preprocess_frame(image, image_cropper, image_bbox, h_input, w_input, scale)
                    start = time.time()
                    prediction += spoof_model.predict(img, os.path.join(model_dir, model_name))
                    test_speed += time.time() - start

                # Draw the prediction result on the image
                label = np.argmax(prediction)
                value = prediction[0][label] / 2

                if label == 1:
                    #print("Real Face. Score: {:.2f}.".format(value))
                    result_text = "RealFace Score: {:.2f}".format(value)
                    S_color = (255, 0, 0)
                    real.append(value)

                else:
                    #print("Fake Face. Score: {:.2f}.".format(value))
                    result_text = "FakeFace Score: {:.2f}".format(value)
                    S_color = (0, 0, 255)
                    fake.append(value)

                #print("Prediction cost {:.2f} s".format(test_speed))
                index = index +  1


            elif  index == spoof_frame_threshold:


                #TODO :Voting for fake or real []

                #1_Normalizing the lists:
                
                

                #2_Get sum of list values
                sum_real = sum(real)
                sum_fake = sum(fake)


                
                #3_Vote if either is larger and set the max value as spoof score
                if sum_fake < sum_real:
                    #real=normalize_list(real)
                    #print(real)
                    spoof_final = 1 - round(max(real),2) #inversing the score of liveness detecion !
                elif sum_fake > sum_real:
                    #fake=normalize_list(fake)
                    #print(fake)
                    spoof_final  = round(max(fake),2)

                    skip_video  =  True

                else:
                    spoof_final = -1


                # print("Final vote for spoof is :",spoof_final)
                final_result[0] = spoof_final



                index = index +  1

            else:
                check_spoof = False
     #------------- Pose Mouth Multi-id ---------#
        
     #-------------------------------------------#
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable=False

        results = face_mesh.process(image)
        face_detection_results = face_detection.process(image)




        image.flags.writeable = True

        image=cv2.cvtColor(image , cv2.COLOR_RGB2BGR)

        img_h , img_w , img_c = image.shape

        face_3d = []
        face_2d = []

        if check_spoof == False:
            counter = counter + 1


        # Check for detected faces (Multi-Faces)
        if face_detection_results.detections:
            num_faces  = len(face_detection_results.detections)
            
            if num_faces > 1:
                final_result[1] = 1
                    
                multi_faces.append(num_faces)
                allow_draw = True

            else :
                allow_draw = False

    

     #---------------- same person --------------#
     # Detect faces in the frame using mediapipe face detection model
        results_for_same_person = face_detection_results

        if results_for_same_person.detections:
            for detection in results_for_same_person.detections:
                # Extract facial landmarks using mediapipe face mesh model
                face_landmarks_results = face_mesh.process(image)
                if face_landmarks_results.multi_face_landmarks:
                    landmarks = face_landmarks_results.multi_face_landmarks[0]

                    # Get distances between selected facial landmarks
                    landmark_points = np.array([(landmark.x, landmark.y, landmark.z ) for i, landmark in enumerate(landmarks.landmark) if i in eye_landmarks + mouth_landmarks])
                    landmark_combinations = np.array([(i,j) for i in range(len(landmark_points)) for j in range(i+1, len(landmark_points))])
                    
                    current_distances = np.array([get_distance(landmark_points[i], landmark_points[j]) for i, j in landmark_combinations])

                    # Compare the distances with the previous frame's distances
                    if previous_distances.any():
                        if any(abs(current_distances[i] - previous_distances[i]) > difference_threshold for i in range(len(current_distances))):
                            # cv2.putText(image, "Different person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            final_result[3] = 1
                        else:
                            # cv2.putText(image, "Same person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            final_result[3] = 0
                    previous_distances = current_distances 
     
    
     #-------------------------------------------#


        # Check for poses (Head and Mouth)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:        
                for idx , lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx ==263 or idx ==1 or idx==61 or idx ==291 or idx ==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h,lm.z *3000)
                        
                        if idx >= 290:
                            if lm.y * img_h > face_2d[1][1]:
                                mouth_open = True
                        x,y = int(lm.x * img_w) , int(lm.y * img_h)

                        face_2d.append([x,y])

                        face_3d.append([x,y,lm.z])
                    
                face_2d = np.array(face_2d , dtype=np.float64)
                face_3d = np.array(face_3d , dtype=np.float64)


                focal_length = 1 * img_w


                cam_matrix = np.array([  [focal_length,0,img_h/2],
                                         [0,focal_length,img_w/2],
                                         [0,0,1]])

                dist_matrix = np.zeros((4,1),dtype = np.float64)


                success , rot_vec , trans_vec  = cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)


                rmat,jac = cv2.Rodrigues(rot_vec)


                angels , mtxR , mtxQ , Qx ,Qy , Qz = cv2.RQDecomp3x3(rmat)


                x= angels[0] * 360
                y= angels[1] * 360
                z= angels[2] * 360
                
                
                # Access lip landmarks
                upper_lip_bottom = face_landmarks.landmark[13].y
                lower_lip_top = face_landmarks.landmark[14].y
                
                # Calculate the distance between upper lip bottom and lower lip top landmarks
                distance = lower_lip_top - upper_lip_bottom


                #Face rotation thresholds
                if x >20:
                    text="UP:6"
                    new_movement=6
                     
                elif x < -20:
                    text="DOWN:7"
                    new_movement=7
                           
                elif y < -14:
                    text="LEFT:8"
                    new_movement=8
                    
                elif y > 14:
                    text="RIGHT:9"
                    new_movement=9
                    
                # Determine if the mouth is open or closed based on the distance
                elif distance > 0.03:  # Adjust this threshold for your specific use case
                    text = "MOUTH OPEN:5"
                    new_movement=5
                    

                #SKIP 2 or 3 Seconds 60 -> 90:TO AVOID ERRORs
                else:
                    text="FORWARD,CENTER"
                    #new_movement = 0

                #sleep(0.35)
                    #Spoof -> 0 1 
                #######
                # face occlusion
                


                   # #----------------------Occlusion---------------------#

                   #  if counter % 30 == 0 and times_OCC_checked < check_OCC_threshold and check_occlussion:
                        
                   #      print("==========================CHECKING FOR OCCLUSION================================")
                   #      detection = face_detection_results.detections[0]
                   #      # Get relative bounding box of that detection            
                   #      boxR = detection.location_data.relative_bounding_box
                   #      ih, iw, _ = image.shape

                   #      # Get Absolute Bounding Box Positions
                   #      # (startX, startY) - Top Left Corner of Bounding Box
                   #      # (endX, endY)     - Bottom Right Corner of Bounding Box
                   #      (startX, startY, endX, endY) = (boxR.xmin, boxR.ymin, boxR.width, boxR.height) * np.array([iw, ih, iw, ih])
                   #      startX = max(0, int(startX))
                   #      startY = max(0, int(startY))
                   #      endX = min(iw - 1, int(startX + endX))
                   #      endY = min(ih - 1, int(startY + endY))

                   #      # Extract the face from the RGB Frame to pass into Mask Detection Model
                   #      face = image[startY:endY, startX:endX]
                   #      face = cv2.resize(face, (224, 224))
                   #      face = img_to_array(face)
                   #      face = preprocess_input(face)
                   #      face = np.array([face], dtype='float32')

                   #      # Predict Mask or No Mask on the extracted RGB Face
                   #      preds = occlusion_detection_model.predict(face, batch_size=32,verbose=None)[0][0]
                   #      label = "No Occlusion" if preds < 0.5 else "Occluded"
                   #      percentage = (1 - preds)  if label == "No Occlusion" else preds 

                   #      if label =="No Occlusion":
                   #          no_occlusion.append((1 - preds))


                   #      else:
                   #          occlusion.append(preds)


                   #      times_OCC_checked = times_OCC_checked + 1


                   #  elif  times_OCC_checked > check_OCC_threshold:
                   #      check_occlussion = False
                   #  else:
                   #      times_OCC_checked += 1
                   # #----------------------------------------------------#
                
                
                
             
                
                #######
                if new_movement != 0  and new_movement != prev_movement:
                    final_result[movement_index] = new_movement
                    detected_movements.append(new_movement)
                    movement_index += 1
                    
                    #print("Movement : ",text)
                    
                    prev_movement=new_movement

                nose_3d_projection ,  jacobian = cv2.projectPoints(nose_3d,rot_vec,trans_vec,cam_matrix,dist_matrix)

                p1 = (int(nose_2d[0]),int(nose_2d[1]))
                p2 = ( int(nose_2d[0] +y *10),int(nose_2d[1] -x *10) )

                ##print(f"p1 is :{p1}/n p2 is :{p2}")
     #-------------------------------------------#
        


        end  = time.time()
        totalTime  = end -start

        #fps = 1/totalTime
     #-------------- Drawing outputs ------------#
        

        cv2.putText(image,f"FPS {int(fps)}",(20,int(img_h-200)),simple_font,1.5,(0,255,0),2)



        # Spoof probiblity
        # if value: 
        #     cv2.putText(image,result_text,(20,150),complex_font, 0.5 * image.shape[0] / 1024, S_color)

        # #Draw Detected Faces
        # if allow_draw: 
        #     #Drawing bounding box around detected faces
        #     for i, detection in enumerate(face_detection_results.detections):
        #         # Extract the bounding box coordinates
        #         bbox = detection.location_data.relative_bounding_box
        #         h, w, c = image.shape
        #         bbox_xmin = int(bbox.xmin * w)
        #         bbox_ymin = int(bbox.ymin * h)
        #         bbox_width = int(bbox.width * w)
        #         bbox_height = int(bbox.height * h)
        #         bbox_xmax = bbox_xmin + bbox_width
        #         bbox_ymax = bbox_ymin + bbox_height

        #         # Get a random color for the bounding box
        #         color = colors[i % len(colors)]

        #         # Draw a rectangle around the face with the assigned color
        #         cv2.rectangle(image, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), color, 2)
        
        #     #Detected faces 
        #     cv2.putText(image, f"Number of Faces: {num_faces}", (20,100), simple_font, 0.7, (0, 0, 255), 2)

        #Face rotation drawings 
        if face_landmarks:
            mp_drawing.draw_landmarks(
                    image= image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
            )

            
            cv2.line(image , p1,p2,(255,0,0),3)
            cv2.putText(image,text,(20,50),simple_font , 2,(0,255,0),2)
            cv2.putText(image,"X : "+str(np.round(x,2)) , (img_w-160,50),simple_font ,1,(0,0,255),2)
            cv2.putText(image,"Y : "+str(np.round(y,2)) , (img_w-160,100),simple_font ,1,(0,0,255),2)
            cv2.putText(image,"Z : "+str(np.round(z,2)) , (img_w-160,150),simple_font ,1,(0,0,255),2)



        #Pop the window
        cv2.imshow("Retroteam FRVT",image)
        if cv2.waitKey(tfps) & 0xFF == ord('q'):#set 27 for Esc key
            break
     #-------------------------------------------#
     #----------------------------------------------#       
            
    cap.release()
    cv2.destroyAllWindows()
    # print("DETECTED  MOVEMENTS:",detected_movements)
    # print(multi_faces)


    #MULTI FACE probiablity needs to be implimented!
    multi_faces = (1 if multi_faces and  (max(multi_faces) > 1) else 0) 
    # final_result[1] = multi_faces

    if len(detected_movements) > 15:
        print("Critial Error in pose estimation please recheck the code")

    #---------------------OUTPUT---------------------#
    occ_final = majority_vote(occlusion,no_occlusion)

    final_result[3] = occ_final
    #------------------------------------------------#

    print(detected_movements)
    # print(final_result)
    return final_result


from glob import glob as g 

for vid in g("./input/*"):
    print(vid)

    main(vid)