import numpy as np
import video_process
import pandas as pd
import os
from datetime import datetime as dt
# from Merged_tasks import main
import time
from tqdm import tqdm
# database address
input = "./input"

lens=len(os.listdir(input))
video_names=np.array(os.listdir(input))

data = [os.path.join(input,vid) for vid in os.listdir(input)]
feature=np.zeros((lens,19))
i=0


for video_name in tqdm(data):    # You should implement the video processing code in this section
    feature[i, :]=video_process.process(video_name,verbose=True)
    # main
    i = i + 1


    
video_names_and_feature=np.hstack((video_names.reshape(lens,1),feature))
Final_csv = pd.DataFrame(np.array(video_names_and_feature))
Final_csv.to_csv('./output/submission.csv', sep=',', index=False, header=False)
print("operation completed successfully")
