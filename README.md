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
prg15<br>
 import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('img.jpg')<br>
image2= cv2.imread('img1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and (image1, image2) <br>
bitwiseor= cv2.bitwise_or(image1, image2)<br>
bitwiseXor=cv2.bitwise_xor (image1, image2)<br>
bitwiseNot_img= cv2.bitwise_not(image1)<br>
bitwiseNot_img1= cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseor)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img) <br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img1)<br>
cv2.waitKey(0)<br><br>
output<br><br>
![download](https://user-images.githubusercontent.com/98144065/176406561-edb928b4-22e2-4ef8-ad76-a38c487a1ca5.png)<br><br>
prg16<br><br>
#importing Libraries<br><br>
import cv2<br><br>
import numpy as np<br><br>
image = cv2.imread('img.jpg')<br><br>
cv2.imshow('Original Image', image)<br><br>
cv2.waitKey(0)<br><br>
# Gaussian Blur<br><br>
Gaussian = cv2.GaussianBlur (image, (7, 7), 0)<br><br>
cv2.imshow('Gaussian Blurring', Gaussian)<br><br>
cv2.waitKey(0)<br><br>
# Median Blur<br><br>
median = cv2.medianBlur (image, 5) <br><br>
cv2.imshow('Median Blurring', median)<br><br>
cv2.waitKey(0)<br><br>
#Bilateral Blur<br><br>
bilateral = cv2.bilateralFilter(image, 9, 75, 75)<br><br>
cv2.imshow('Bilateral Blurring', bilateral)<br><br><br>
cv2.waitKey(0)<br><br><br>
cv2.destroyAllWindows()<br><br><br>
output
![Screenshot 2022-06-29 151750](https://user-images.githubusercontent.com/98144065/176406949-426e6b74-4f68-4388-a021-c87f70df726f.png)<br><br><br>
![image](https://user-images.githubusercontent.com/98144065/176420860-423a6414-0e98-4929-a1dc-ea322bf22f40.png)<br>
![image](https://user-images.githubusercontent.com/98144065/176420944-b73b53f5-9ebf-451c-bee4-7481607cf881.png)<br>
![image](https://user-images.githubusercontent.com/98144065/176421020-54810ea7-b5bc-4f39-a14a-f1c51a25c0da.png)<br>
prg17<br><br><br>
from PIL import Image<br><br><br>
from PIL import ImageEnhance <br><br><br>
image =Image.open('img3.jpg')<br><br><br>
image.show()<br><br><br>
enh_bri =ImageEnhance.Brightness(image)<br><br><br>
brightness= 1.5<br><br><br>
image_brightened= enh_bri.enhance(brightness)<br><br><br>
image_brightened.show()<br><br><br>
enh_col= ImageEnhance.Color(image)<br><br><br>
color= 1.5<br><br><br>
image_colored =enh_col.enhance(color)<br><br><br>
image_colored.show()<br><br><br>
enh_con =ImageEnhance.Contrast (image) <br><br><br>
contrast = 1.5<br><br><br>
image_contrasted =enh_con.enhance(contrast)<br><br><br>
image_contrasted.show()<br><br><br>
enh_sha =ImageEnhance.Sharpness(image)<br><br><br>
sharpness =3.0<br><br><br>
image_sharped= enh_sha. enhance (sharpness)<br><br><br>
image_sharped.show()<br><br><br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/176408175-c205dea9-5523-42d2-b983-2ab0c1a1f54f.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/176408301-99d7a350-3e45-4162-b747-d3f8ad83617d.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/176408349-19ecfef1-5200-4023-b2b0-8bbb7a6dfb2f.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/176408411-72919615-66e6-4fec-ae1e-276bd4fd1605.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/176408466-02597098-409a-4fee-b3ea-a2ec359e08d8.png)<br><br>
prg 18<br><br>
import cv2<br><br>
import numpy as np<br><br>
from matplotlib import pyplot as plt<br><br>
from PIL import Image, ImageEnhance<br><br>
img= cv2.imread('img3.jpg',1)<br><br>
ax=plt.subplots(figsize=(20,10))<br><br>
kernel = np.ones((5,5), np.uint8)<br><br>
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) <br><br>
closing=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)<br><br>
erosion= cv2.erode(img,kernel,iterations=1)<br><br>
dilation=cv2.dilate(img,kernel,iterations=1)<br><br>
gradient=cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)<br><br>
plt.subplot (151)<br><br>
plt.imshow(opening)<br><br>
plt.subplot (152)<br><br>
plt.imshow(closing)<br><br>
plt.subplot(153)<br><br>
plt.imshow(erosion)<br><br>
plt.subplot(154)<br><br>
plt.imshow(dilation)<br><br>
plt.subplot (155)<br><br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/176410428-bac5374f-f904-41ff-98b6-ba94f9770576.png)<br>
prg19<br>
import cv2<br>
OriginalImg=cv2.imread('img1.jpg')<br>
GrayImg=cv2.imread('img1.jpg',0)<br>
isSaved=cv2.imwrite('D:/i.jpg', GrayImg) <br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image', GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is successfully saved.')<br>
    output<br>
    ![image](https://user-images.githubusercontent.com/98144065/178697184-d50e099a-ac16-4d3c-b57f-4949809bf185.png)<br>
    ![image](https://user-images.githubusercontent.com/98144065/178697229-e18e7262-89fe-4695-94ae-15c208c52455.png)<br>
    ![image](https://user-images.githubusercontent.com/98144065/178697455-3f7b717a-5a58-4066-a53c-365faac250d0.png)<br>\
    prg20<br>
    import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('img1.jpg', 0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack ((image, z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/178699646-b6f11c22-f112-41d3-9a21-afefef6f6030.png)<br>
prg21<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('img1.jpg',0) <br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in  range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image, z))<br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/178705138-38301cd4-7a05-43e4-98bb-3474b1e8ac66.png)<br>
<br>
prg22<br><br><br>

import cv2<br><br>

from matplotlib import pyplot as plt<br><br>

img = cv2.imread('img.jpg',0)<br><br>
plt.imshow(img)<br><br>
plt.show()<br><br>

histr = cv2.calcHist([img],[0],None,[256],[0,256])<br><br>
plt.plot(histr)<br><br>
plt.show()<br><br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/178959806-52df6dfd-f137-41b9-bbde-bea26e80a452.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/178960240-ee12c05f-9d92-4bc4-94d5-c9dc80689792.png)<br><br>

prg23<br><br>
import cv2<br><br>
import numpy as np<br><br>
img  = cv2.imread('img1.jpg',0)<br><br>
plt.imshow(img)<br><br>
plt.show()<br><br>
hist = cv2.calcHist([img],[0],None,[256],[0,256])<br><br>
plt.hist(img.ravel(),256,[0,256])<br><br>

plt.show()<br><br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/178959925-61c9a3b6-ab5a-4ff3-84bd-9504af104f80.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/178960309-a4791935-6950-4386-b2f0-63f639eaa1f2.png)<br><br>
prg24<br><br>
Program to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings ("ignore", category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('img1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179945068-88514aab-e250-4b5c-b082-e548264073f6.png)
![image](https://user-images.githubusercontent.com/98144065/179947690-79f0a173-5756-4a23-83a1-df1fb000e5d0.png)<br><br>

prg25<br>
negative =255- pic #neg = (L-1) img <br>
plt.figure(figsize= (6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179945355-bee6e5d0-710e-4c64-b48a-d7b878680f6c.png)<br><br>

prg26<br><br>
%matplotlib inline<br>
import imageio <br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('img1.jpg') <br>
gray=lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587,0.114]) <br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(), cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179948638-b4d41d22-5899-487d-a2ef-b14d60ee944c.png)<br>
<br>
prg27<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
#Gamma encoding<br>
pic=imageio.imread('img3.jpg')<br>
gamma=2.2 #Gamma < 1 ~ Dark; Gamma > 1~ Bright<br>

gamma_correction=((pic/255)**(1/gamma)) <br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/179949121-b3a06a89-a30b-41e4-b8fa-bd8ba86b633a.png)<br><br>
prg28<br><br>
Program to perform basic image manipulation:<br>
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>
from PIL import Image<br><br>
from PIL import ImageFilter <br><br>
import matplotlib.pyplot as plt<br><br>
 
my_image = Image.open('img3.jpg')<br><br>
sharp= my_image.filter(ImageFilter.SHARPEN)<br><br>

sharp.save('D:/i2.jpg')<br><br>
sharp.show() <br><br>
plt.imshow(sharp)<br><br>
plt.show()<br><br>
output<br><br>
![image](https://user-images.githubusercontent.com/98144065/179951827-c47ff40b-92de-40c4-96de-0e636ada6b2d.png)<br><br>
![image](https://user-images.githubusercontent.com/98144065/179952229-0dc22517-dcf6-4fd4-9b9b-fda9e4c1e24a.png)<br><br>
prg29<br><br>
#Image flip<br><br>
import matplotlib.pyplot as plt<br><br>
#Load the image<br><br>
img = Image.open('img3.jpg')<br><br>
plt.imshow(img) <br><br>
plt.show()<br><br>
#use the flip function<br><br>
flip = img.transpose(Image.FLIP_LEFT_RIGHT)<br><br>
#save the image<br><br>
flip.save('D:/image_flip.jpg')<br><br>
plt.imshow(flip)<br>
plt.show()<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/179952438-e40c8837-ea1c-4903-9a43-5ea663ee8e2a.png)<br>
![image](https://user-images.githubusercontent.com/98144065/179952464-a9a98c5d-4c30-46a9-98d3-6ae2eaa5a44c.png)<br>
![image](https://user-images.githubusercontent.com/98144065/179952614-67099e11-9ec2-4e6c-ba86-601daf7a80fe.png)<br>

prg30<br>
# Importing Image class from PIL module <br>
import matplotlib.pyplot as plt # Opens a image in RGB mode <br>
im=Image.open('img3.jpg')<br>
# Size of the image in pixels (size of original image) #(This is not mandatory)<br>
width, height = im.size<br>
#Cropped image of above dimension # (It will not change original image) <br>
im1=im.crop ((280, 100, 800, 600))<br>
#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98144065/179952541-b0492478-04de-4903-835b-693f1d7c49fc.png)<br>
PRG31<br>
import cv2<br>
img = cv2.imread('img4.jpg')<br>
cv2.imshow('original', img)<br>
cv2.waitKey(0)<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
![image](https://user-images.githubusercontent.com/98144065/186653044-03cc5a3f-8d3f-4495-8103-1bf1c229eb46.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186653108-f5370fe4-279b-4a60-b156-acf62e90b00d.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186653145-e650f808-cc29-48d0-9b1e-d3aad6b3a255.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186653168-47b24ad3-b172-40c4-acee-94c5e0edfa15.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186653184-450694a0-f341-419f-b8ad-34079a003611.png)
PRG33<br>
import matplotlib.pyplot as plt<br>
%matplotlib inline<br>
from skimage import data,filters<br>
image = data.coins()<br>
edges = filters.sobel(image)<br>
plt.imshow(edges, cmap='gray')<br>
OUTPUT<br>
<matplotlib.image.AxesImage at 0x2ee3db2e520><br>
![image](https://user-images.githubusercontent.com/98144065/186653411-5019909e-e3b6-46fe-b202-5069567df0a4.png)<br>
PRG34<br>
from PIL import Image, ImageChops, ImageFilter<br>
from matplotlib import pyplot as plt<br>
#Create a PIL Image objects<br>
x= Image.open("x.png") <br>
o=Image.open("o.png")<br>
#Find out attributes of Image Objects<br>
print('size of the image: ', x.size,' colour mode:', x.mode) <br>
print('size of the image: ', o.size ,'colour mode:', o.mode)<br>
#plot 2 images one besides the other<br>
plt.subplot(121), plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122), plt.imshow(o)<br>
plt.axis('off')<br>
#multiply images<br>
merged=ImageChops.multiply(x,o)<br>
#adding 2 images <br>
add =ImageChops.add(x,o)<br>
#convert colour mode<br>
greyscale=merged.convert('L')<br>
greyscale<br>
OUTPUT<br>
size of the image:  (257, 255)  colour mode: RGBA<br>
size of the image:  (259, 256) colour mode: RGBA<br>
![image](https://user-images.githubusercontent.com/98144065/186653509-f59eb2c5-0309-4dc6-9fce-dbdc9eda17cd.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186653513-32f67d79-3435-40d7-82e3-bac45bdca80b.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186653530-f0b29d6e-b6c5-41fd-9d8c-4f6d0c7d1e90.png)<br>
PRG35<br>
image=merged<br>
print('image size: ',image.size,<br>
'\ncolor mode: ',image.mode,<br>
'\nimage width: ', image.width,'| also represented by: I also represented by:' ,image.size[0],<br>
'\nimage height:', image.height,'| also represented by: I also represented by:' ,image.size[1],)<br>
OUTPUT<br>
image size:  (257, 255) <br>
color mode:  RGBA <br>
image width:  257 | also represented by: I also represented by: 257 <br>
image height: 255 | also represented by: I also represented by: 255<br>

 #mapping the pixels of the image so we can use them as coordinates<br>
pixel=greyscale.load()<br>
#a nested Loop to parse through all the pixels in the image<br>
for row in range(greyscale.size[0]): <br>
    for column in range(greyscale.size[1]): <br>
        if pixel[row, column] != (255):<br>
            pixel[row, column] = (0)<br>
greyscale<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/98144065/186653670-a5a0a439-ee8f-4d12-8f83-473187147ba9.png)<br>
#1. invert image<br>
invert = ImageChops.invert(greyscale)<br>
#2. invert by subtraction<br>
bg=Image.new('L', (256, 256), color=(255)) #create a new image with a solid white background <br>
subt=ImageChops. subtract (bg, greyscale) #subtract image from background<br>
#3. rotate<br>
rotate = subt.rotate(45)<br>
rotate<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/98144065/186653755-8ed1dd60-7707-4666-80ab-66eb9ac033a4.png)<br>
#gaussian blur<br>
blur= greyscale.filter(ImageFilter.GaussianBlur (radius=1))<br>
#edge detection.<br>
edge=blur.filter(ImageFilter.FIND_EDGES) <br>
edge<br>
![image](https://user-images.githubusercontent.com/98144065/186653835-1837d47f-1646-4a93-8d78-51b4fde95f4d.png)<br>
#change edge colours<br>
edge=edge.convert('RGB')<br>

bg_red=Image.new('RGB', (256,256), color=(255,0,0))<br>
filled_edge = ImageChops.darker(bg_red, edge)<br>
filled_edge<br>
![image](https://user-images.githubusercontent.com/98144065/186653902-e5d7abeb-9719-41e9-8e79-0f9e04b88378.png)<br>
#save image in the directory <br>
edge.save('processed.png')<br>
PRG36<br>
import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
#Open the image.<br>
img=cv2.imread('cat_damaged.png')<br> 
plt.imshow(img)<br>
plt.show()<br>
#Load the mask.<br>
mask=cv2.imread('cat_mask.png', 0)<br>
plt.imshow(mask)<br>
plt.show()<br>
# Inpaint.<br>
dst=cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)<br>
# write the output.<br>
cv2.imwrite('cat_inpainted.png', dst)<br>
plt.imshow(dst)<br>
plt.show()<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/98144065/186654061-aae8b3f0-822f-4393-aced-568b7fb6d0bc.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654088-091b04bd-8d87-4b28-b327-abb527b34746.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654118-daf598f6-1cfc-4989-a1a5-23f9c576d86a.png)<br>
PRG37<br>
import numpy as np<br>
import pandas as pd<br>
import matplotlib.pyplot as plt <br>
plt.rcParams['figure.figsize'] = (10, 8)<br>
def show_image(image, title='Image', cmap_type='gray'):<br>
    plt.imshow(image, cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
def plot_comparison(img_original, img_filtered, img_title_filtered):<br>
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)<br> 
    ax1.imshow(img_original, cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered, cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br>
    from skimage.restoration import inpaint <br>
from skimage.transform import resize<br>
from skimage import color<br>
image_with_logo = plt.imread('imlogo.png')<br>
                             
# Initialize the mask <br>
mask = np.zeros(image_with_logo.shape[:-1])<br>
# Set the pixels where the Logo is to 1<br>
mask[210:272, 360:425] = 1<br>
# Apply inpainting to remove the Logo<br>
image_logo_removed=inpaint.inpaint_biharmonic (image_with_logo,mask, multichannel=True)<br>
# show the original and Logo removed images<br>
plot_comparison (image_with_logo, image_logo_removed, 'Image with logo removed')<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/98144065/186654339-a3a8d98b-988f-4aa4-a843-8e50ffd54612.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654377-bff94fc9-7084-4290-aa31-66cef2a73fd8.png)<br>
from skimage.util import random_noise<br>
fruit_image = plt.imread('fruitts.jpeg')<br>
# Add noise to the image<br>
noisy_image = random_noise (fruit_image)<br>
#Show th original and resulting image<br>
plot_comparison (fruit_image, noisy_image, 'Noisy image')<br>
![image](https://user-images.githubusercontent.com/98144065/186654465-be561a66-7956-43d6-98da-47d918974376.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654488-be228e6b-d05e-4efd-b26e-fda55b2215fc.png)<br>
from skimage.restoration import denoise_tv_chambolle<br>
noisy_image = plt.imread('noisy.jpg')<br>
# Apply total variation filter denoising<br>
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)<br>
#Show the noisy and denoised image <br>
plot_comparison (noisy_image, denoised_image, 'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/98144065/186654558-2b953203-d721-40d2-a88f-5f744d052d14.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654594-8568ebd7-b346-4690-aa1e-abc81b5a711a.png)<br>
from skimage.restoration import denoise_bilateral<br>
landscape_image = plt.imread('noisy.jpg')<br>
# Apply bilateral filter denoising <br>
denoised_image = denoise_bilateral (landscape_image, multichannel=True)<br>
# Show original and resulting images<br>
plot_comparison (landscape_image, denoised_image, 'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/98144065/186654646-7ec45703-230a-47ce-a8d3-f2605a122636.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654677-b4e84a97-60ee-414e-ac66-a5c0f958bfe1.png)<br>

from skimage.segmentation import slic <br>
from skimage.color import label2rgb<br>
face_image = plt.imread('face.jpg')<br>
# obtain the segmentation with 400 regions<br>
segments=slic(face_image, n_segments=400)<br>
# Put segments on top of original image to compare<br>
segmented_image = label2rgb(segments, face_image, kind='avg')<br>
# Show the segmented image<br>
plot_comparison(face_image, segmented_image, 'Segmented image, 400 superpixels')<br>
![image](https://user-images.githubusercontent.com/98144065/186654739-4efdce3e-a8fb-4a31-a709-137080e2ce2c.png)<br>
![image](https://user-images.githubusercontent.com/98144065/186654762-acaae099-1d3a-4682-8c33-563dc68b72c4.png)<br>
prg37<br>
def show_image_contour (image, contours): <br>
    plt.figure()<br>
    for n, contour in enumerate(contours):<br>
        plt.plot(contour[:,1], contour[:,0 ], linewidth=3)<br>
    plt.imshow(image, interpolation='nearest', cmap='gray_r')<br>
    plt.title("Contours")<br>
    plt.axis('off')<br>
    from skimage import measure, data<br>
#obtain the horse image <br>
horse_image = data.horse()<br>
# Find the contours with a constant Level value of 0.8<br>
contours = measure.find_contours (horse_image, level=0.8)<br>
# Shows the image with contours found <br>
show_image_contour (horse_image, contours)<br>
output<br>
 ![image](https://user-images.githubusercontent.com/98144065/186654982-43790b3d-f182-4366-a630-2741cc9bf60e.png)<br>
prg38<br>
from skimage.io import imread <br>
from skimage.filters import threshold_otsu<br>
image_dices = imread('diceimg.png')<br>
# Make the image grayscale<br>
image_dices = color.rgb2gray(image_dices)<br>
# obtain the optimal thresh value<br>
thresh = threshold_otsu(image_dices)<br>
# Apply thresholding<br>
binary=image_dices > thresh<br>
# Find contours at a constant value of 0.8 <br>
contours = measure.find_contours(binary, level=0.8)<br>
#Show the image<br>
show_image_contour (image_dices, contours)<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/186655059-b09ed085-c300-44bf-bbca-cc35100198b1.png)<br>
prg39<br>
# Create List with the shape of each contour <br>
shape_contours = [cnt.shape[0] for cnt in contours]<br>
# Set 58 as the maximum size of the dots shape<br>
max_dots_shape = 50<br>
#count dots in contours excluding bigger than dots size <br>
dots_contours = [cnt for cnt in contours if np. shape (cnt) [0] < max_dots_shape]<br>
#Shows all contours found <br>
show_image_contour (binary, contours)<br>
#Print the dice's number<br>
print('Dices dots number: .'.format(len (dots_contours)))<br>
output<br>
Dices dots number: .<br>
![image](https://user-images.githubusercontent.com/98144065/186655196-6749f7a0-398c-4571-9d57-24dafbc6bf44.png)<br>






    













