from detectCars import *

import solve

img = mpimg.imread('../test_images/test6.jpg')
#img = mpimg.imread('vlcsnap-28.jpg')
solve.saveRGB('35-in.jpg', img)

bboxes = detect_cars(img)

for bbox in bboxes:
    cv2.rectangle(img,(bbox[0][0], bbox[0][1]),(bbox[1][0], bbox[1][1]),(0,0,255),6) 

solve.saveRGB('35-out.jpg', img)

#plt.imshow(out_img)

with open('bbox_pickle.p', 'wb') as output:
    pickle.dump(bboxes, output, pickle.HIGHEST_PROTOCOL)
