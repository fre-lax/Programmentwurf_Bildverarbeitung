import numpy as np
import cv2
from ffmpeg import FFmpeg
import shutil
import os

def cosCalc(x,y,w,h,beta):
    xm=(x-w/2)
    ym=(y-h/2)
    abs=np.sqrt(xm**2+ym**2)
    abs=abs**0.6
    angle=np.arctan2(xm,ym)
    return (np.cos(abs*2+angle+beta)+1)*127.49

def runX(image, result,settings=None):
    height, width, = image.shape[:2]  # image height and width
    imageGray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY); 

    image2=image.copy()
    for y in range(height): 
        for x in range(width):
            image2[y,x]=image[height-y-1,x]
    result.append({"name":"yFlip","data":image2})
    result.append({"name":"yFlip2","data":np.flip(image,0)})
    result.append({"name":"yFlip3","data":cv2.flip(image2,0)})
    
    image3=image.copy()
    for y in range(height):
        for x in range(width):
            image3[y,x]=255-image[y,x]
    result.append({"name":"Invert","data":image3})
    result.append({"name":"Invert2","data":255-image})

    image4=image2.copy()
    for y in range(height):
        for x in range(width):
            image4[y,x]=255-imageGray[y,x]
    result.append({"name":"InvertGray","data":image4})
    result.append({"name":"InvertGray2","data":255-image4})

    #Hier werden Bilder erzeugt und zu einem Video zusammengefasst.
    #Die Videoerstellung war nicht Teil der Aufgabe
    #Für die Videoerzeugung ist ffmpeg notwendig. Dazu das Paket installieren mit  "pip install python-ffmpeg".
    #Zudem muss "ffmpeg.exe" (https://ffmpeg.org/download.html#build-windows) in den Winpython Ordner "t".
    os.makedirs("tmp", exist_ok=True)
    for i in np.arange(0,360,3):
        beta=i/np.pi/2
        w,h = 100,200
        cosMat = np.fromfunction(lambda x,y:cosCalc(x,y,w,h,beta), (w, h), dtype=int)
        cv2.imwrite("tmp\output"+str(int(i/3)).rjust(3, '0')+".png",cosMat)

    ffmpeg = ( FFmpeg().option("y").option("r",18)
        .input("tmp\output%03d.png")
        .output(
            "output.mp4",
            {"codec:v": "libx264"}
        )
    )
    ffmpeg.execute()
    shutil.rmtree("tmp")
    
if __name__ == '__main__':
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    runX(image,result)
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()