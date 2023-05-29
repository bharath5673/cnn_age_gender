import cv2
import numpy as np
import yolov5
from keras.models import load_model

#Load age and gender model
model_path = "./model.h5"
model = load_model(model_path)

# Load the YOLOv5 model
# model_path = ' yolov5s.pt'
model_path = 'crowdhuman_yolov5m.pt'


# device = "cpu"  # for cpu
device = 0  #for gpu
yolov5 = yolov5.YOLOv5(model_path,device,load_on_init=True)

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




# Process each frame of the video
while True:
  # Read the next frame
  success, frame = video.read()
  if not success:
    break


  # Perform object detection on the frame
  results = yolov5.predict(frame, size = 640, augment=False)
  detections = results.pred[0]


  # Check whether the bounding box centroids are inside the ROI
  for detection in detections:    
    xmin    = detection[0]
    ymin    = detection[1]
    xmax    = detection[2]
    ymax    = detection[3]
    score   = detection[4]
    class_id= detection[5]
    centroid_x = int(xmin + xmax) // 2
    centroid_y =  int(ymin + ymax) // 2

    #Threshold score
    if score >= 0.3:  
      if class_id == 0:
        color = (255, 0, 0)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
      else:
        color = (0, 0, 225)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

        age_ = []
        gender_ = []
        face = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        # cv2.imwrite("test1.jpg",face)
        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img,(200,200))
        predict = model.predict(np.array(img).reshape(-1,200,200,3))
        age_.append(predict[0])
        gender_.append(np.argmax(predict[1]))
        gend = np.argmax(predict[1])
        if gend == 0:
          gend = 'Male'
        else:
          gend = 'Female'
          
        col = (0,224,0)
        cv2.putText(frame,"Age : "+str(int(predict[0])),(int(xmin),int(ymax)),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,2)
        cv2.putText(frame,"Gender : "+str(gend),(int(xmin),int(ymax+15)),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,2)



  # Display the frame
  cv2.imshow("Video", frame)
  video_writer.write(frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break



# Release the video capture object
video.release()
video_writer.release()

cv2.destroyAllWindows()
