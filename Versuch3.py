import numpy as np
import cv2

def run(image, result,settings=(0,0)):
    if(len(image.shape)!=3):
            print("Nur für Farbbilder")
            return
    height, width, _ = image.shape[:]  # image height and width

    image2=np.zeros((height,width,3),np.uint8)
    b, g, r = cv2.split(image)
    mask = (b <77) & (g >90) & (r >98)
    image2[mask] = (30,120,170) 
    mask = (b <56) & (g <86) & (r >94)
    image2[mask] = (20,50,170) 
    mask = (b <86) & (g <120) & (r <20)
    image2[mask] = (80,110,10) 
    mask = (b >50) & (g <55) & (r <20)
    image2[mask] = (120,40,10) 
    mask = (b <35) & (g <100) & (r <100)
    image2[mask] = (30,100,90) 
    result.append({"name":"Abgleich","data":image2})
    print(cv2.COLOR_BGR2GRAY)
    image3=image.copy()
    _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),135,255,cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    conList=[]

    for i,ele in enumerate(contours):
        if 2000>cv2.contourArea(ele)>300:
            conList.append(ele)
            color=cv2.cvtColor(np.uint8([[(i*10,255,255)]]), cv2.COLOR_HSV2BGR_FULL)
            color=(int(color[0,0,0]),int(color[0,0,1]),int(color[0,0,2]))
            cv2.fillPoly(image3,[ele],color)

    #cv2.fillPoly(image3,conList,(200,10,212))
    cv2.drawContours(image3, conList, -1, (0,255,0), 1)
    result.append({"name":"Abgleich123","data":image3})


    

if __name__ == '__main__':
    image=cv2.imread("Images\Farbpunkte.jpg")
    result=[]
    run(image,result)
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()