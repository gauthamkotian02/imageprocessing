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
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.show()
<br>
output<br>
![image](https://user-images.githubusercontent.com/98144065/174053401-8c74c2fd-f4b1-4303-a0b0-8e311d14ed18.png)
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

