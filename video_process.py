import cv2
import torch
import mediapipe as mp
import time
import os
import sys
import random
import warnings
import numpy as np
from datetime import datetime as dt

#Our Imports
import gdown
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

#Spoof detection imports
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

#Base line imports

from torchvision.transforms import functional as F
from facenet_pytorch import MTCNN,InceptionResnetV1


warnings.filterwarnings('ignore')




def download_file(file_id, output_path):
    try:
        gdown.download(file_id, output_path, quiet=False)
        print("File downloaded successfully!")
    except Exception as e:
        print(f"Error occurred while downloading the file: {e}")

# Example usage
# file_url = "https://example.com/path/to/model.h5"
# output_file_path = "model.h5"


#============================= Define Once =============================#

#---------------------------- Constatns ----------------------------#
#Number of GPUS for Pytorch
device_id = 0


gdown_lst = {"vggface2.pt":"1VijRF3CZhRHTd0ea8U4ZsxIkUMlZWmUX",
             "occlusion_detection_model.h5":"10EKrw08j1o8pWXWGXVMnyqbsrpKrjDsz"}



#-------------------- Face Mesh/Detection  cv2 cascades setup --------------------#
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence =0.5 ,min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence =0.4) 


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255, 118, 0), thickness=1,circle_radius =1)

eye_landmarks = [1, 4, 5, 246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133, 247, 30, 29, 27, 28, 56, 190, 130, 25, 110, 24, 23, 22, 26, 112, 243, 113, 225]

mouth_landmarks = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146,91, 181, 84, 17, 314, 405]

difference_threshold = 0.099


# Load pre-trained Haar Cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained Haar Cascade classifier for detecting eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')



#------------------------- model load ------------------------#
spoof_model = AntiSpoofPredict(device_id)
image_cropper = CropImage()
model_dir="./models/anti_spoof_models"
occ_model_path = "./models/occlusion_detection_model.h5"


try:
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()#Same person
    face_ids = set()
    #model = load_model('./models/model.h5')#Check mask
    occlusion_detection_model  = load_model(occ_model_path)
except Exception as e:
    print("Model files haven't been loaded correctly redownloading now")
    print(e.args)
    for k,v in gdown_lst.items():
        print(f"downloading {k}")
        download_file(id=v,output_path=f"./models/{k}")  
finally:
    print("Reloading models now...")
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()#Same person
    face_ids = set()
    #model = load_model('./models/model.h5')#Check mask
    occlusion_detection_model  = load_model(occ_model_path)

#=====================================Functions===============================#
#Function  to preprocess the frame for spoof
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

def detect(videoname,verbose=False):
    tic = time.time()


 ################################################# Varibles ###############################################
    # Initialize variables
    index               = 0
    mask_count          = 0
    acculotion          = 0
    glasses_count       = 0
    count               = 0
    movement_counter    = 0
    movement_index      = 4
    threshold           = 0.6
    frame_rate          = 10

    spoof_frame_threshold     = 10
    occlusion_frame_threshold = 5
    occlusion_frame           = 0


    total_result        = []
    multi_face_result   = []
    mask_prob           = []
    detected_movements  = []
    fake                = []
    real                = []
    spoof_dict = {}

    eyes                    = None
    new_movement            = None
    prev_movement           = None
    label                   = None
    allow_draw              = None
    face_landmarks          = None
    spoof_final             = None
    previous_face_encodings = None
    
    check_spoof      = True
    check_occlussion = True

    multi_checked    = False
    skip_video       = False


    final_result = np.zeros(19)
    previous_distances = np.array([])

    cap = cv2.VideoCapture(videoname)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)




    while True:
        ret, frame = cap.read()
        multi_face = 0


        #if video finished
        if not ret:
            break


        count += 1

        #if frame rate wasn't desired 
        if count % frame_rate != 0:
            continue
        #Video was spoof and no lognger check for stuff!
        if skip_video ==True:

            final_result[0] = spoof_final
            final_result[1:] = 0
            break
            #all other tasks remain  0 -> multi_face,acculotion,multi_id
            #return  final_result
 #############################################  spoof detect  #############################################|TODO:1
        if check_spoof:
            if  index < spoof_frame_threshold:
                # Preprocess the image
                image_bbox = spoof_model.get_bbox(frame)
                prediction = np.zeros((1, 3))
                test_speed = 0


                # Perform prediction using each model
                for model_name in os.listdir(model_dir):
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    img = preprocess_frame(frame, image_cropper, image_bbox, h_input, w_input, scale)
                    start = time.time()
                    prediction += spoof_model.predict(img, os.path.join(model_dir, model_name))
                    test_speed += time.time() - start

                # Draw the prediction result on the frame
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
 #########################################  Head and Mouth Rotation  ######################################|TODO:2
        #Video isn't spoof so we continue to do other tasks
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)


        img_h , img_w , img_c = frame_rgb.shape
        face_3d = []
        face_2d = []
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

                #print(new_movement)
                #Face rotation thresholds
                if x >18:
                    text="UP:6"
                    new_movement = 6
                     
                elif x < -14:
                    text="DOWN:7"
                    new_movement = 7
                           
                elif y < -14:
                    text="LEFT:8"
                    new_movement = 8
                    
                elif y > 14:
                    text="RIGHT:9"
                    new_movement = 9
                    
                # Determine if the mouth is open or closed based on the distance
                elif distance > 0.02:  # Adjust this threshold for your specific use case
                    text = "MOUTH OPEN:5"
                    new_movement=5
                

                #SKIP 2 or 3 Seconds 60 -> 90:TO AVOID ERRORs
                else:
                    text="FORWARD,CENTER"

                

                    #OTHER OPERATIONS GO HERE!
 ##########################################  multiface & multi id  ########################################
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect faces in frame
                boxes, _ = mtcnn.detect(frame_rgb)
                if boxes is not None:
                    multi_face_result.append(len(boxes))


                # # Loop through detected faces
                if boxes is not None:
                    for box in boxes:
                        # Extract face from frame
                        face = F.to_tensor(frame_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
                        if (face.shape[1] >= 160 & face.shape[2] >= 160):
                            face = F.resize(face, (160, 160))

                            face = torch.unsqueeze(face, dim=0).float()
                            # Normalize face
                            face = (face - 127.5) / 128.0
                            face = face.cuda()

                            # Calculate face embeddings using InceptionResnetV1 model
                            embeddings = resnet(face)

                            # Convert embeddings to numpy array
                            embeddings_np = embeddings.detach().cpu().numpy()

                            # Convert embeddings to string
                            embeddings_str = embeddings_np.tostring()

                            # Calculate face ID based on embeddings
                            face_id = hash(embeddings_str)

                            # Check if face ID is new
                            if face_id not in face_ids:
                                # Add face ID to set
                                face_ids.add(face_id)
 ###############################################   Occlusion  ################################################|TODO:Test my model
  ####################################   keras inception resnet model   ###################################
                frame = frame_rgb


                if check_occlussion == True and occlusion_frame_threshold > occlusion_frame:


                    face_detection_results = face_detection.process(frame)
                    detection = face_detection_results.detections[0]
                    ih, iw, _ = frame.shape
                    boxR = detection.location_data.relative_bounding_box
                    # Get Absolute Bounding Box Positions
                    # (startX, startY) - Top Left Corner of Bounding Box
                    # (endX, endY)     - Bottom Right Corner of Bounding Box
                    (startX, startY, endX, endY) = (boxR.xmin, boxR.ymin, boxR.width, boxR.height) * np.array([iw, ih, iw, ih])
                    startX = max(0, int(startX))
                    startY = max(0, int(startY))
                    endX = min(iw - 1, int(startX + endX))
                    endY = min(ih - 1, int(startY + endY))

                    # Extract the face from the RGB Frame to pass into Mask Detection Model
                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.array([face], dtype='float32')

                    # Predict Mask or No Mask on the extracted RGB Face
                    preds = occlusion_detection_model.predict(face, batch_size=32,verbose=None)[0][0]
                    occlusion_frame = occlusion_frame+1
                    if preds >0.5:
                        final_result[3]=1
                        check_occlussion = True
  ############################################ mask detection ############################################
                # Preprocess frame
                # frame = frame_rgb #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (224, 224))
                # frame = preprocess_input(frame)

                # # Predict mask probability
                # pred = model.predict(np.expand_dims(frame, axis=0),verbose=None)[0]#,verbose=None
                # mask_prob = pred[0]
                # no_mask_prob = pred[1]

                # # Check if wearing a mask
                # if mask_prob > no_mask_prob:
                #     mask_count += 1
  ############################################### sunglasses ##############################################
                # faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # # Loop through detected faces
                # for (x, y, w, h) in faces:
                # # Extract face region of interest
                #     face_roi = frame_rgb[y:y + h, x:x + w]

                # # Detect eyes in face region
                #     eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                #     # Check if wearing glasses
                #     if len(eyes) >= 2:
                #         glasses_count += 1


                #Multi face
                for item in multi_face_result:
                    if item > 1:
                        final_result[1]=1
                    else:
                        final_result[1]=0
                        
                #Same person
                if len(face_ids)>1:
                    final_result[3]=1
                else:
                    final_result[3]=0

                #acculotion
                if mask_count>0 and glasses_count>0:
                    final_result[2]=1



                        


            #######
            #logic -> if previous movement and new movements aren't same
            #######
            if new_movement and  new_movement != prev_movement and len(detected_movements) <=15:


                final_result[movement_index] = new_movement
                detected_movements.append(new_movement)
                movement_index += 1
                
                #print("Movement : ",text)
                
                prev_movement=new_movement
 ################################################## Output ###############################################
    #final_result[0:3] -> Spoof [0], multi_face[1],acculotion[2],multi_id[3]

    #final_result[3:19] -> movements

    spoof       = round(final_result[0]) 
    multi_face  = final_result[1]
    acculotion  = final_result[2]
    multi_id    = final_result[3]
    movements   = final_result[4:]


    if verbose:
        print(f"""
        ====================={videoname}====================

        Time       : {time.time() - tic} seconds
        
        spoof      : {spoof}

        multi_face : { multi_face}

        acculotion : {acculotion}

        multi_id : {multi_id }

        movements: { movements}
        ===================================================""")
    
    return (spoof , multi_face , acculotion , multi_id , movements)

def process(videoname,verbose=False):
    result = []

    #final_result[0:3] -> Spoof [0], multi_face[1],acculotion[2],multi_id[3]
    #final_result[3:19] -> movements

    spoof , multi_face , acculotion , multi_id , movements =detect(videoname,verbose)
    result.append(spoof)     # 0 or 1
    result.append(multi_face)# 0 or 1
    result.append(acculotion)# 0 or 1
    result.append(multi_id)  # 0 or 1


    #Head rotations
    my_list = movements      #[0] * 15
    result.extend(my_list)

    return result
