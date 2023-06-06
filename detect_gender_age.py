# -*- coding: utf-8 -*-

#DeepFace lib is required , install it by running => pip install deepface

import requests
from deepface import DeepFace

def exec_test():
  file_url = "https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcTYiQcFe-WmL2c9zefm1O1aDnRzz5QuEUK06F-bYAQdALalZjlGiCZkKp9GvV3crsSaJtofK-xziOe7hso"
  filename="brad.jpeg"
  response = requests.get(file_url)
  if response.status_code == 200:
    with open(filename, "wb") as file:
      file.write(response.content)
    return True
  else:
    return False

def age_gender(img):
  res=DeepFace.analyze(img)
  if res:
    result = dict(res[0])
    print("\n","Detected age :",result["age"] ,"\n","Detected Gender :",result["dominant_gender"])
    return  result["age"] , result["dominant_gender"]

exec_test()
age , gender = age_gender("./brad.jpeg")







