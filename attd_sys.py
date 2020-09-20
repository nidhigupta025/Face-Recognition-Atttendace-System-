import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime
import csv

path="Students"
images=[]
names=[]


mylist=os.listdir(path)
#print(mylist)

for student in mylist:
    curimg=cv2.imread(f'{path}/{student}')
    images.append(curimg)
    names.append(os.path.splitext(student)[0])
#print(names)

def find_encoding(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return  encodelist


filename = "university_records.csv"
def markattendance(name):
    with open(filename,'r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}') 


#markattendance('Nidh')

enocdelistknown=find_encoding(images)
print(len(enocdelistknown))

cap=cv2.VideoCapture(0)

while True:
    success, img= cap.read()
    #img reduced to 1/4th of its real size for faster processing
    img_s=cv2.resize(img,(0,0),None,0.25,0.25)   
    #converting BGR image to RGB image
    img_s=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #faces_in_frame is an array containing the co-ordinates of each face in the frame
    faces_in_frame=face_recognition.face_locations(img_s)
    #list of encodings for each face in the frame
    encode_cur_frame=face_recognition.face_encodings(img_s,faces_in_frame)
    
    for encodeface, faceloc in zip(encode_cur_frame,faces_in_frame):
        matches=face_recognition.compare_faces(enocdelistknown,encodeface)
        facedist=face_recognition.face_distance(enocdelistknown,encodeface)
        #print(facedist)
        matchindex=np.argmin(facedist)
        if matches[matchindex]:
            name=names[matchindex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)
    cv2.imshow('Webcam image',img)
    if cv2.waitKey(1) == ord('q'):
        break
    
#cv2.waitKey(1)
cv2.destroyAllWindows()
cap.release()
                       
        


