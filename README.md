# imageprocessing
prg1<br>
<br>
import cv2
img=cv2.imread('img.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
<br>
output<br>![image](https://user-images.githubusercontent.com/98144065/174052236-47923c00-554a-41af-9776-29c038312cd5.png)

prg2<br>
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('img.jpg')
plt.imshow(img)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174052514-53996dcc-1424-49b8-8631-843b4874da45.png)
<br>
prg3<br>
from PIL import Image
img=Image.open("img.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174052705-89c7b113-d4ba-4387-ad7d-7f92785c89b4.png)
<br>
prg4<br>
from PIL import ImageColor
img=ImageColor.getrgb("red")
print(img)<br>
output<br>
(255, 0, 0)<br>
prg5<br>
from PIL import ImageColor
img=Image.new('RGB',(200,400),(254,150,100))
img.show()
<br>
output<br>
<br>
![image](https://user-images.githubusercontent.com/98144065/174053024-642efebe-0e8f-4769-b6e4-7d75ed5cafdd.png)
<br>
prg6<br>
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('img.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()
<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174058127-096ea256-6e39-4646-819b-ee048fb84b89.png)

<br>
prg7<br>
from PIL import Image
image=Image.open('img.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()
<br>
output<br>
Filename: img.jpg
Format: WEBP
Mode: RGB
Size: (737, 480)
Width: 737
Height: 480
<br>
prgt8<br>
import cv2
img=cv2.imread('img1.jpg')
print('originally image length width',img.shape)
cv2.imshow('original image',img)
cv2.waitKey(0)
imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize)
print('Resized image length width',imgresize.shape)
cv2.waitKey(0)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174053925-eedac977-542c-4e38-af49-7abb9990436e.png)
![image](https://user-images.githubusercontent.com/98144065/174054013-e886db9f-4469-4c0a-8469-813b5d285854.png)
<br>
prg9 rgb,grey,threshold<br>
import cv2
img=cv2.imread('img.jpg')
cv2.imshow("RGB",img)
cv2.waitKey(0)
img=cv2.imread('img.jpg',0)
cv2.imshow("Gray",img)
cv2.waitKey(0)
ret, bw_img=cv2.threshold(img,127,100,cv2.THRESH_BINARY)
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174056384-124dfe7e-68aa-4499-a3ed-f3f219ce9baa.png)
![image](https://user-images.githubusercontent.com/98144065/174056421-9f24489c-0724-4619-9577-08839509a24e.png)
![image](https://user-images.githubusercontent.com/98144065/174056456-eaf01bfb-422b-4f1c-9a7b-bacf3fbfb937.png)<br>
prg10<br>
from skimage import io
import matplotlib.pyplot as plt
url='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()<br>
outout<br>
![image](https://user-images.githubusercontent.com/98144065/175263156-ac858d70-2716-4b85-8c7f-81528cd46748.png)<br>
prg11<br>
import cv2
import matplotlib.image as mping 
import matplotlib.pyplot as plt
img1=cv2.imread('img1.jpg')
image1=cv2.imread('image1.jpg')
fimg1=img1+image1
plt.imshow(fimg1)
plt.show()<br>
![image](https://user-images.githubusercontent.com/98144065/175263796-60d2ed5e-352a-40ed-9648-04531fd7b6fe.png)<br>
fimg1=img1-image1
plt.imshow(fimg1)
plt.show()<br>
![image](https://user-images.githubusercontent.com/98144065/175263975-15f0fb51-6f70-4c90-9f37-232cce48fa9b.png)
<br>
fimg1=img1*image1
plt.imshow(fimg1)
plt.show()
![image](https://user-images.githubusercontent.com/98144065/175264064-c3ee50f7-d55b-4943-8f60-8a143546e747.png)
<br>
fimg1=img1/image1
plt.imshow(fimg1)
plt.show()
<br>
prg12
import cv2
import matplotlib.image as mping 
import matplotlib.pyplot as plt 
img=mping.imread('image3.jpg') 
plt.imshow(img) 
plt.show()
<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175269805-6070c278-88a5-4c48-a677-3ddef68cf9d9.png)
<br>
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
light_orange = (1,190,200)
dark_orange=(18,255,255)
mask = cv2.inRange(hsv_img,light_orange,dark_orange) 
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175270078-686520df-fe45-4f21-a799-2732dc8ab68c.png)
<br>
light_white=(0, 0, 200)
dark_white=(145, 60, 255)
mask_white=cv2.inRange(hsv_img, light_white, dark_white)
result_white=cv2.bitwise_and(img, img, mask=mask_white)
plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175270268-6f2e91df-c7d2-44cd-9583-21e00f8bff5f.png)<br>
final_mask=mask+mask_white
final_result = cv2.bitwise_and(img, img, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()
<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175271487-9e0f7423-0e79-4e05-af53-0c969464e440.png)
blur=cv2.GaussianBlur(final_result, (7, 7), 0)
plt.imshow(blur)
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175271621-7d31b2ef-bdcb-4df1-b2f6-1b12c9f4c70e.png)<br>
prg13<br>
import cv2 
img = cv2.imread("D:\image1.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV) 
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls) 
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175272332-07154bdb-f5b3-48c0-b08f-fdf27ddd8450.png)
<br>
prg14<br>
import cv2 as c
import numpy as np
from PIL import Image
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100]=[255, 130, 0]
array[:,100:]=[0, 0, 255]
img = Image.fromarray(array)
img.save('image1.jpg')
img.show()
c.waitKey(0)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175272609-89692f11-d452-49ec-bbdc-c5a5f3c5ea67.png)










