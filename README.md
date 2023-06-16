# Face Cup Competition code  Documentation 2023
<br>


![alt text](https://mms.businesswire.com/media/20210406005123/en/869329/5/AdobeStock_397652837_IDEMIA_light.jpg)
![alt text](https://facecup.ir/wp-content/uploads/2022/12/Face-Cup_Animation_1.gif)

This documentation provides an overview of the Face Cup competition and its seven tasks for facial recognition based on the FRVT (Facial Recognition Vendor Test) standard. The 

instructions below will guide you on how to set up and run the competition tasks using Docker and the provided models.



## Setup



#### Input :
a list of videos inside ./input folder

#### output:
A csv file within ./output folder containing a list of different detetions,which is explained bellow


To run the Face Cup competition, follow the steps below:
<ol>
	<li>Ensure that you have access to Colab for testing and Docker for deployment.</li>
	<li>Download the following files from Google Drive and copy them to the models folder:</li>
	<ul>
		<li>
	Vgg and model.h5 for Person reid and mask detection: <a href="https://drive.google.com/drive/folders/19TQxvYLXsEQc4dGGAaNslRNz0Y3Esced" alt="google drive link for the model">Google Drive link</a> 
			</li>
 <li>
	Occlusion detection model: <a href="https://drive.google.com/file/d/10EKrw08j1o8pWXWGXVMnyqbsrpKrjDsz/edit" alt="google drive link for the model">Google Drive link</a>
</li>
		 <li><b>Please note that you need this models downloaded and moved to models dir in order for the program to run properly</b></li>
	 </ul>



<li>
	
Create a folder named input and output and paste all your videos into input folder 

</li>

</ol>


Install requirements using :
 ```
pip install -r requirements.txt
```


To run the program using :

```
python run.py
```


## Competition Details
The Face Cup competition consists of the following tasks:


### Task 1: Spoof Detection
<ul>
<li>Run limited:</li>
<li>Spoof: 0 - 1 [✓]</li>
</ul>

### Task 2: Face Occlusion Detection
<ul>
<li>Run in Loop:</li>
<li>Face occlusion: 0 - 1 [✓]</li>
</ul>

### Task 3: Multi-Face Identification
<ul>
<li>Run in Loop:</li>
<li>Multi-face ID: 0 - 1 [✓]</li>
</ul>


### Task 4: Mouth and Head Movement Detection

<ul>
<li>Run in Loop:</li>
<li>Mouth and head movement: [✓]</li>
</ul>

### Task 5: Reid Verification

<ul>
<li>Run in Loop:</li>
<li>Person Reid: 0 - 1 [✓]</li>
</ul>

### Task 6: Multi-Identity Recognition

<ul>
<li>Run in Loop:</li>
<li>Multi ID: 0 - 1 [TEST]</li>
</ul>

### Task 7: Capabilities

<ul>
The system is capable of performing the following tasks:
<li>Pose estimation: [✓]</li>
<li>Mouth detection: [✓]</li>
<li>Counting: [✓]</li>
<li>Spoof detection: [✓]</li>
<li>Same person verification: [✓]</li>
<li>Face occlusion detection: [✓]</li>
</ul> 



## Optimization and Evaluation Metrics


The following optimizations can be implemented:
<ul>
<li>Simple Deploy Docker [✓]</li>
<li>Change Code Architecture [✓]</li>
<li>Change Code Blocks [✓]</li>
<li>Set Frame [✓]</li>
<li>Change Model [✓]</li>
</ul>

By following these instructions and using the provided models, you can participate in the Face Cup competition and evaluate your facial recognition system's performance against the specified tasks.


The evaluation metrics for the competition tasks are as follows:
<ul>
<li>Scores: 0.3 (Spoof), 0.2 (Multi-Faces), 0.2 (Occlusion), 0.1 (Multi-Id), 0.2 (Movements [6:20])</li>
<li>P_time = 20 * (1 - runtime/180)</li>
<li>Accuracy = 80 * (avg(modul_scores))</li>
</ul>




## About Us
We as Retroteam were able to compete and achieve  4th in this compettion among  more than 300 teams, with an accuracy of 98 percent and 48 minutes of runtime processing 400 videos.
This competition was originally held by nextra startup


## Contact
Feel free to contact us via email or connect with us on linkedin.

- Elyas Moshirpanahi --- [Linkedin](https://www.linkedin.com/in/ElyasMoshirpanahi1997), [Github](https://github.com/ElyasMoshirpanahi), [Email](mailto:elyasmoshirpanahe1376@gmail.com)
- Pourya Bahmanyar --- [Linkedin](https://www.linkedin.com/in/pouria-bahmanyar-20b2201b8/), [Github](https://github.com/pouria-bahmanyar), [Email](mailto:pooriabahman@gmail.com)


