# Face Cup Competition code  Documentation 2023
<br>


![alt text](https://mms.businesswire.com/media/20210406005123/en/869329/5/AdobeStock_397652837_IDEMIA_light.jpg)
![Uploading Face-Cup_Animation_1.gif…](https://facecup.ir/wp-content/uploads/2022/12/Face-Cup_Animation_1.gif)

This documentation provides an overview of the Face Cup competition and its seven tasks for facial recognition based on the FRVT (Facial Recognition Vendor Test) standard. The 

instructions below will guide you on how to set up and run the competition tasks using Docker and the provided models.



## Setup

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
	 </ul>


 <li>Please note that you need this models downloaded and moved to models dir in order for the program to run properly</li>
</ol>

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
