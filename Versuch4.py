import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

def run(image, result,settings=(0,0)):
    if(len(image.shape)==3):
        imageGray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY); 
    else:
        imageGray=image
    height, width = imageGray.shape[:]  # image height and width

    _, thresh = cv2.threshold(imageGray,135,255,cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    segment=[]
    for idx,ele in enumerate(contours):
        if(cv2.contourArea(ele)<10): continue
        x,y,w,h=cv2.boundingRect(ele)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(image,str(idx)+" "+str(cv2.contourArea(ele)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0))

        M = cv2.moments(ele)
        cX,cY = int(M["m10"] / M["m00"]) , int(M["m01"] / M["m00"])
        cv2.circle(image,(cX,cY),1,(255,255,0),)
        bins=90
        arr=np.ones(bins)*(-1)
        for ele2 in ele:
            y,x=ele2[0][1]-cY,ele2[0][0]-cX
            r=np.sqrt(y**2+x**2)
            a=int((np.arctan2(y,x)/np.pi+1)/2*(bins-0.00001))
            arr[a]=r

        arr2=arr.copy()
        for j in range(0, 5):
            arr2=(np.roll(arr2, 1)+np.roll(arr2, -1)+arr2)/3

        arr2sP,arr2sM=np.roll(arr2, 1),np.roll(arr2, -1)
        corners=np.count_nonzero((arr2>arr2sP) & (arr2>=arr2sM)) #np.sum wäre auch möglich

        result.append({"name":str(idx)+" "+str(corners),"data":arr})
        result.append({"name":str(idx)+" "+str(corners),"data":arr2})

    image=cv2.resize(image,None,None,3,3)
    result.append({"name":"Result","data":image})

    
if __name__ == '__main__':
    matplotlib.use('Agg')
    image=cv2.imread("Images\Objekte.png")
    result=[]
    run(image,result)
    for ele in result:
        if(len(ele["data"].shape)==1):
            fig=plt.figure(num=ele["name"])
            plt.plot(ele["data"])
            fig.tight_layout()
            plt.grid(True)
            plt.xlim(0.0, len(ele["data"])-1)
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer._renderer)
            cv2.imshow(ele["name"],data)
        else:
            cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()