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
