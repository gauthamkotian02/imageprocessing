# imageprocessing<br>
prg1<br><br>
<br>
import cv2<br>
img=cv2.imread('img.jpg',0)<br>
cv2.imshow('image',img)
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
output<br>![image](https://user-images.githubusercontent.com/98144065/174052236-47923c00-554a-41af-9776-29c038312cd5.png)<br>

prg2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('img.jpg')<br>
plt.imshow(img)<br><br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174052514-53996dcc-1424-49b8-8631-843b4874da45.png)<br>
<br>
prg3<br><br>
from PIL import Image<br>
img=Image.open("img.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/174052705-89c7b113-d4ba-4387-ad7d-7f92785c89b4.png)<br>
<br>
prg4<br>
from PIL import ImageColor<br>
img=ImageColor.getrgb("red")<br>
print(img)<br>
output<br><br>
(255, 0, 0)<br>
prg5<br>
from PIL import ImageColor<br>
img=Image.new('RGB',(200,400),(254,150,100))<br>
img.show()<br>
<br>
output<br>
<br>
![image](https://user-images.githubusercontent.com/98144065/174053024-642efebe-0e8f-4769-b6e4-7d75ed5cafdd.png)<br>
<br>
prg6<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('im<br>g.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
<br><br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174058127-096ea256-6e39-4646-819b-ee048fb84b89.png)<br>

<br>
prg7<br>
from PIL import Image<br>
image=Image.open('img.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>
<br>
output<br>
Filename: img.jpg<br>
Format: WEBP<br>
Mode: RGB<br>
Size: (737, 480)<br>
Width: 737<br>
Height: 480<br>
<br>
prgt8<br>
import cv2<br>
img=cv2.imread('img1.jpg')<br>
print('originally image length width',img.shape)<br><br><br>
cv2.imshow('original image',img)<br><br><br>
cv2.waitKey(0)<br><br><br>
imgresize=cv2.resize(img,(150,160))<br><br><br>
cv2.imshow('Resized image',imgresize)<br><br><br>
print('Resized image length width',imgresize.shape)<br><br><br>
cv2.waitKey(0)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174053925-eedac977-542c-4e38-af49-7abb9990436e.png)<br><br><br>
![image](https://user-images.githubusercontent.com/98144065/174054013-e886db9f-4469-4c0a-8469-813b5d285854.png)<br><br><br>
<br>
prg9 rgb,grey,threshold<br><br><br><br>
import cv2
img=cv2.imread('img.jpg')<br><br><br>
cv2.imshow("RGB",img)<br><br><br>
cv2.waitKey(0)<br><br><br>
img=cv2.imread('img.jpg',0)<br><br><br>
cv2.imshow("Gray",img)<br><br><br>
cv2.waitKey(0)<br><br><br>
ret, bw_img=cv2.threshold(img,127,100,cv2.THRESH_BINARY)<br><br><br>
cv2.imshow("Binary",bw_img)<br><br><br>
cv2.waitKey(0)<br><br><br>
cv2.destroyAllWindows()<br><br><br><br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174056384-124dfe7e-68aa-4499-a3ed-f3f219ce9baa.png)<br><br><br>
![image](https://user-images.githubusercontent.com/98144065/174056421-9f24489c-0724-4619-9577-08839509a24e.png)<br><br><br>
![image](https://user-images.githubusercontent.com/98144065/174056456-eaf01bfb-422b-4f1c-9a7b-bacf3fbfb937.png)<br><br><br><br>
prg10<br><br><br><br>
from skimage import io<br><br><br>
import matplotlib.pyplot as plt<br><br><br>
url='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.jpg'<br><br><br>
image=io.imread(url)<br><br><br>
plt.imshow(image)<br><br><br>
plt.show()<br><br><br><br>
outout<br>
![image](https://user-images.githubusercontent.com/98144065/175263156-ac858d70-2716-4b85-8c7f-81528cd46748.png)<br>
prg11<br><br><br>
import cv2<br><br>
import matplotlib.image as mping <br><br>
import matplotlib.pyplot as plt<br><br>
img1=cv2.imread('img1.jpg')
image1=cv2.imread('image1.jpg')<br><br>
fimg1=img1+image1<br><br>
plt.imshow(fimg1)<br><br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98144065/175263796-60d2ed5e-352a-40ed-9648-04531fd7b6fe.png)<br>
fimg1=img1-image1<br><br>
plt.imshow(fimg1)<br><br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98144065/175263975-15f0fb51-6f70-4c90-9f37-232cce48fa9b.png)<br><br>
<br>
fimg1=img1*image1<br><br>
plt.imshow(fimg1)<br><br>
plt.show()
![image](https://user-images.githubusercontent.com/98144065/175264064-c3ee50f7-d55b-4943-8f60-8a143546e747.png)<br>
<br>
fimg1=img1/image1<br>
plt.imshow(fimg1)<br>
plt.show()<br>
<br><br>
prg12<br>
import cv2<br>
import matplotlib.image as mping <br>
import matplotlib.pyplot as plt <br>
img=mping.imread('image3.jpg') <br>
plt.imshow(img) <br>
plt.show()<br>
<br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/175269805-6070c278-88a5-4c48-a677-3ddef68cf9d9.png)<br>
<br>
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)<br>
light_orange = (1,190,200)<br>
dark_orange=(18,255,255)<br>
mask = cv2.inRange(hsv_img,light_orange,dark_orange) <br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br><br><br>
plt.imshow(mask, cmap="gray")<br><br><br>
plt.subplot(1,2,2)<br><br><br>
plt.imshow(result)<br><br><br>
plt.show()<br><br><br><br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175270078-686520df-fe45-4f21-a799-2732dc8ab68c.png)<br><br><br>
<br>
light_white=(0, 0, 200)<br><br><br>
dark_white=(145, 60, 255)<br><br><br>
mask_white=cv2.inRange(hsv_img, light_white, dark_white)<br><br><br>
result_white=cv2.bitwise_and(img, img, mask=mask_white)<br><br><br>
plt.subplot(1, 2, 1)<br><br><br>
plt.imshow(mask_white, cmap="gray")<br><br><br>
plt.subplot(1, 2, 2)<br><br><br>
plt.imshow(result_white)<br><br><br>
plt.show()<br><br><br><br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175270268-6f2e91df-c7d2-44cd-9583-21e00f8bff5f.png)<br>
final_mask=mask+mask_white<br><br>
final_result = cv2.bitwise_and(img, img, mask=final_mask)<br><br>
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")<br><br>
plt.subplot(1, 2, 2)<br><br>
plt.imshow(final_result)<br><br>
plt.show()<br><br>
<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175271487-9e0f7423-0e79-4e05-af53-0c969464e440.png)<br><br>
blur=cv2.GaussianBlur(final_result, (7, 7), 0)<br><br>
plt.imshow(blur)<br><br>
plt.show()<br>
output<br><br><br>
![image](https://user-images.githubusercontent.com/98144065/175271621-7d31b2ef-bdcb-4df1-b2f6-1b12c9f4c70e.png)<br><br><br>
prg13<br><br><br>
import cv2 <br><br>
img = cv2.imread("D:\image1.jpg")<br><br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV) <br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls) <br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/175272332-07154bdb-f5b3-48c0-b08f-fdf27ddd8450.png)<br>
<br>
prg14<br>
import cv2 as c<br>
import numpy as np
from PIL import Image<br>
array = np.zeros([100, 200, 3], dtype=np.uint8)<br>
array[:,:100]=[255, 130, 0]<br>
array[:,100:]=[0, 0, 255]<br>
img = Image.fromarray(array)<br>
img.save('image1.jpg')<br>
img.show()<br>
c.waitKey(0)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/175272609-89692f11-d452-49ec-bbdc-c5a5f3c5ea67.png)<br>










