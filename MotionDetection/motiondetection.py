
import cv2 as cv 
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
sender_mail='123@gmail.com'
reciever_mail='2010080040@gmail.com'
password='riyaa'

cap = cv.VideoCapture(0) 
kernel=np.ones((5,5),np.uint8)
background_subtractor = cv.createBackgroundSubtractorMOG2()
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    ret, frame = cap.read()
    if not ret:
        break
    fg_mask = background_subtractor.apply(frame) 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_TRIANGLE)
    contours, hierarchy = cv.findContours(binary, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    
    draw_con = cv.drawContours(frame, contours, -1, (255, 255, 255), thickness=2, lineType=cv.LINE_AA)
    draw_con=cv.morphologyEx(draw_con,cv.MORPH_OPEN,kernel)  
    
    motion_detected = bool(len(faces))
    
    for (a,b,c,d) in faces:
        cv.rectangle(draw_con,(a,b),(a+c,b+d),(0,0,255),thickness=2)
    for i,(a,b,c,d) in enumerate(faces):
        frames=draw_con[a:a+c,b:b+d]
        frames=draw_con[b:b+d,a:a+c]
        saved_image=f"faces{i}.jpg" 
        cv.imwrite(saved_image,frames)
    if motion_detected==True:
        cv.putText(draw_con, f'{len(faces)} people ALERT Motion Detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    if motion_detected!=True:
        subject='RED ALERT'
        message='Motion Detected in your house/office'
        msg = MIMEMultipart()
        msg['From'] = sender_mail
        msg['To'] = reciever_mail
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain')) 
    
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_mail, password)
            server.sendmail(sender_mail, reciever_mail, msg.as_string())
            server.quit()
            print("Email alert sent!")
        except Exception as e:
            print("Failed to send email alert:", str(e))


    cv.imshow('Motion Detection', draw_con)

    if cv.waitKey(6) == 27:  
            break
    

cap.release()
cv.destroyAllWindows()
