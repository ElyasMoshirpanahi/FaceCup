<img src='[](https://facecup.ir/wp-content/uploads/2022/12/Face-Cup_Animation_1.gif)'>


#Need 2 files from google drive copied to models folder to run properly!
Download each at :
<br>
Vgg and model.h5 for Person reid and mask detection : https://drive.google.com/drive/folders/19TQxvYLXsEQc4dGGAaNslRNz0Y3Esced
<br>
Occlusion detection model at : https://drive.google.com/file/d/10EKrw08j1o8pWXWGXVMnyqbsrpKrjDsz/edit
# Face cup competetion 2023

![alt text](https://mms.businesswire.com/media/20210406005123/en/869329/5/AdobeStock_397652837_IDEMIA_light.jpg)
![alt text](https://facecup.ir/wp-content/uploads/2022/12/Face-Cup_Animation_1.gif)


____________________________________________________________________________

Video Name | Spoof | Multi-Faces | Occlusion | Multi-Id | Movements [6:20] |

Scores	   |  0.3  |	  0.2    |    0.2    |	  0.1   |        0.2       |
____________________________________________________________________________

P_time = 20 * (1 - runtime/180)<br>
Accuracy = 80 * (avg(modul_scores))<br>
____________________________________________________________________________

[0 - 4]<br>
Run limited:<br>
	/t Spoof     0 - 1    [✓] <br>

	
[5- 20]<br>
Run in Loop:<br>
	Face occulasion  0 - 1   [✓] <br>
	multi face id  0 - 1  [✓]<br>
	mouth and head movment [✓] <br>
	same person  0 - 1     [✓] <br>
	multi id  0 - 1  [TEST]<br>

____________________________________________________________________________
Capeable of the following tasks:<br>
	<br>pose        [✓] 
	<br>mouth       [✓] 
	<br>count       [✓] 
	<br>Spoof       [✓] 
	<br>Same person [✓]
	<br>Occlusion   [✓]
	
____________________________________________________________________________
To optimize:
	<br>Simple Deploy Docker     [✓]
	<br>Change Code Architecture []
	<br>Change code blocks       []		
	<br>Set frame			     []
	<br>Change model             [] 
	<br>Turn into OOP            []
	<br>Runtime optimizations    []
	
___________________________________________________________________________