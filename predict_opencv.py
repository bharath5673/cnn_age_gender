import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

#Load age and gender model
model_path = "./model.h5"
model = load_model(model_path)


# Load the video
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture("test.mp4")


# Get the video's width, height, and frames per second (fps)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
# Create a VideoWriter object to save the video
output_file = 'output_video.mp4'  # Specify the output video file name
video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Process each frame of the video
while True:
  success, frame = video.read()
  if not success:
    break

  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray,1.3,5)
  age_ = []
  gender_ = []
  for (x,y,w,h) in faces:
    img = gray[y-50:y+40+h,x-10:x+10+w]
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img,(200,200))
    predict = model.predict(np.array(img).reshape(-1,200,200,3))
    age_.append(predict[0])
    gender_.append(np.argmax(predict[1]))
    gend = np.argmax(predict[1])
    if gend == 0:
      gend = 'Male'
    else:
      gend = 'female'

    col = (0,224,0)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,224),2)
    cv2.putText(frame,"Age : "+str(int(predict[0])),(int(x),int(y+h)),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,2)
    cv2.putText(frame,"Gender : "+str(gend),(int(x),int(y+h+15)),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,2)



  # Display the frame
  cv2.imshow("Video", frame)
  video_writer.write(frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break



# Release the video capture object
video.release()
video_writer.release()

cv2.destroyAllWindows()
