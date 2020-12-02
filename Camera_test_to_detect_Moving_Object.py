import cv2
import time             #Helpful for the time delay
import imutils 
cam = cv2.VideoCapture(0)  #Initialize the camera
time.sleep(1)

firstFrame=None          #Fix the first frame
area = 500               #helpful for taking as the benchmark & identifying to how much extent is the change
count=0
while True:
    _,img = cam.read()   #Read the camera feed
    text = "Normal"     #Initialize the text
    img = imutils.resize(img, width=500)  # Resize for better image view
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conversion of BGR to Gray
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) # For smoothening and threshold purpose
    if firstFrame is None:
            firstFrame = gaussianImg
            continue      # To pass the code to  further functions
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)  # To identify the diff. btw. the current frame(Gaussian Img) and the first frame(BackGround Img)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #Thresholding is to change the pixels of an image to make the image easier to analyze
                       # Thresholding is the  Image segmentation that isolates objects by converting grayscale images into binary images.
    threshImg = cv2.dilate(threshImg, None, iterations=2)   # Dilate is to expand the border thickness of the image.Erosion is to thin down the border thickness
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,  
            cv2.CHAIN_APPROX_SIMPLE)         # contours for finding the neighbourhood pixels and connect them to form a complete image
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
            if cv2.contourArea(c) < area:
                    continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  #To draw the rectangle, if once the object is detected 
            count+=1
            text = "Moving Object detected"+str(count)
    print(text)
    cv2.putText(img, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    #To display the text once the object is detected
    cv2.imshow("cameraFeed",img)    
    key = cv2.waitKey(1) & 0xFF     # Waitkey represents the 32 bit value . 0xFF represents 8 bit value
    if key == ord("q"):     # Termination
        break        # Breaks the while True function
    count=0
cam.release()  #To release the camera
cv2.destroyAllWindows()
