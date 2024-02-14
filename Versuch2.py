import numpy as np
import cv2

def run(image, result,settings=None):
    height, width, *_ = image.shape[:]  # image height and width
    b,g,r=cv2.split(image)

    minB,maxB=cv2.minMaxLoc(b)[0:2]
    minG,maxG,_,_=cv2.minMaxLoc(g)
    minR,maxR,_,_=cv2.minMaxLoc(r)
    print(maxB,maxG,maxR,minB,minG,minR)

    b=(b-minB)/(maxB-minB)
    g=(g-minG)/(maxG-minG)
    r=(r-minR)/(maxR-minR)

    image2=cv2.merge((b,g,r))
    result.append({"name":"jeder Kanal getrennt","data":image2})

    max=np.max((maxB,maxG,maxR))
    min=np.min((minB,minG,minR))
    
    image3=(image-min)/(max-min)
    result.append({"name":"alle Kanäle gleich","data":image3})
    

    imageGray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    imageGray[int(height/10):9*int(height/10),int(height/10):2*int(height/10)]=\
        np.fromfunction(lambda y,x: y/(8*int(height/10))*255.9,(8*int(height/10),int(height/10)))
    #Pixelzugriff als Alternative (kommt man einfacher drauf ist aber langsamer)
    #for y in range(int(height/10),9*int(height/10)):
    #    for x in range(int(height/10),2*int(height/10)):
    #        imageGray[y,x]=(1-((y-int(height/10))/(8*int(height/10))))*255
    result.append({"name":"Graukeil","data":imageGray})

    r=(-np.sin(imageGray/255.9*2*np.pi)+1)/2
    g=(-np.cos(imageGray/255.9*2*np.pi)+1)/2
    b=(np.sin(imageGray/255.9*2*np.pi)+1)/2

    result.append({"name":"Pseudo","data":cv2.merge((b,g,r))})

    image4=image.copy()
    _,test,_,maxLoc=cv2.minMaxLoc(cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY))
    max=image4[maxLoc[1],maxLoc[0]]
    print(f'max: {max}')
    print(f'test: {test}')
    print(maxLoc,max)

    image4[:,:,0]=image4[:,:,0]/max[0]*240
    image4[:,:,1]=image4[:,:,1]/max[1]*240
    image4[:,:,2]=image4[:,:,2]/max[2]*240
    result.append({"name":"Abgleich","data":image4})


if __name__ == '__main__':
    image=cv2.imread("graphics/heli.png")
    result=[]
    run(image,result)
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()