import numpy as np 
from PIL import Image , ImageOps
import math
import cv2

from Function_Ai import *

# image geometry transformation  


# scale / size transformation

# reflection

# rotation

#  shear 


#----------------------------- scale   تبدیلات هندسی 

# img = Image.open('a2.jpg')   # 815 733
# img_size = img.size
# width, height = img_size
# im_gray = ImageOps.grayscale(img)
# pixels = list(im_gray.getdata())
# array = np.array(pixels,dtype='uint8').reshape(height, width)

#######################################  روش معمولی 
# for i in range(height):
#     for j in range(width):
#         point= np.array([i,j])
#         scale_point = scale_array.dot(point)
#         s_i , s_j = scale_point[0], scale_point[1]
#         new_array[s_i,s_j] = array[i,j]

####################################### 2x

# array ,height , width = covert_to_array (image='a2.jpg')
# scale_array = np.array([ [2, 0],
#                         [0, 2]])
# new_array = np.ones((2 *  height , 2 * width ) , dtype='uint8') * 255   # ساختن یه آرایه خالی و ضرب در 255 کردن چون میخوایم بک گراند سفید باشه 

# scale = np.linalg.inv(scale_array)
# for s_i in range(height * 2):
#     for s_j in range(width * 2):
#         s_point = np.array([s_i, s_j])
#         scale_point = scale.dot(s_point)
#         i, j = scale_point[0], scale_point[1]
#         i = int(math.floor(i))
#         j = int(math.floor(j))
#         new_array[s_i, s_j] = array[i, j]

# new_image = Image.fromarray(new_array)
# new_image.save('s.jpg')


#####################################  varone x

# array ,height , width = covert_to_array (image='a2.jpg')
# scale_array = np.array([ [1, 0],
#                         [0, -1]])
# new_array = np.ones((height , width ) , dtype='uint8') * 255   # ساختن یه آرایه خالی و ضرب در 255 کردن چون میخوایم بک گراند سفید باشه 

# scale = np.linalg.inv(scale_array)
# for s_i in range(height):
#     for s_j in range(width):
#         s_point = np.array([s_i, s_j])
#         scale_point = scale.dot(s_point)
#         i, j = scale_point[0], scale_point[1]
        
#         new_array[s_i, s_j] = array[height - s_i - 1, s_j]

# new_image = Image.fromarray(new_array)
# new_image.show()

##################################### varnone y

# array ,height , width = convert_to_array (image='a2.jpg')
# scale_array = np.array([ [-1, 0],
#                         [0, 1]])
# new_array = np.ones((height , width ) , dtype='uint8') * 255   # ساختن یه آرایه خالی و ضرب در 255 کردن چون میخوایم بک گراند سفید باشه 

# scale = np.linalg.inv(scale_array)
# for s_i in range(height):
#     for s_j in range(width):
#         s_point = np.array([s_i, s_j])
#         scale_point = scale.dot(s_point)
#         i, j = scale_point[0], scale_point[1]
        
#         new_array[s_i, s_j] = array[ s_i , width - s_j - 1]

# new_image = Image.fromarray(new_array)
# new_image.show()


################################ varone xy

# array ,height , width = convert_to_array (image='a2.jpg')
# scale_array = np.array([ [-1, 0],
#                         [0, -1]])
# new_array = np.ones((height , width ) , dtype='uint8') * 255   # ساختن یه آرایه خالی و ضرب در 255 کردن چون میخوایم بک گراند سفید باشه 

# scale = np.linalg.inv(scale_array)
# for s_i in range(height):
#     for s_j in range(width):
#         s_point = np.array([s_i, s_j])
#         scale_point = scale.dot(s_point)
#         i, j = scale_point[0], scale_point[1]
        
#         new_array[s_i, s_j] = array[ height - s_i - 1 , width - s_j - 1]

# new_image = Image.fromarray(new_array)
# new_image.show()



#------------------------------------------------      دوران دادن عکس 

# array, height, width = convert_to_array(image='a2.jpg')



# rotation_matrix = np.array([
#     [np.cos(90), -np.sin(90)],  
#     [np.sin(90), np.cos(90)]    
# ])

# print(rotation_matrix)

# new_height = int(height * np.abs(np.cos(45)) + width * np.abs(np.sin(45)))
# new_width = int(height * np.abs(np.sin(45)) + width * np.abs(np.cos(45)))
# new_array = np.ones((new_height, new_width), dtype='uint8') * 255  # ایجاد پس‌زمینه سفید

# scale = np.linalg.inv(rotation_matrix)
# r = height**2 + width**2
# r2 = math.sqrt(r)
# for s_i in range(height):
#     for s_j in range(width):
#         s_point = np.array([s_i, s_j])
#         scale_point = rotation_matrix.dot(s_point)
#         i, j = scale_point[0], scale_point[1]
        
#         new_array[s_i, s_j] = array[i  , width - j  - 1]

# new_image = Image.fromarray(new_array)
# new_image.show()
# #---------
# array, height, width = convert_to_array(image='a2.jpg')

# angle_radians = np.deg2rad(90)

# rotation_matrix = np.array([
#     [np.cos(angle_radians), -np.sin(angle_radians)],  
#     [np.sin(angle_radians), np.cos(angle_radians)]    
# ])

# new_height = int(height * np.abs(np.cos(angle_radians)) + width * np.abs(np.sin(angle_radians)))
# new_width = int(height * np.abs(np.sin(angle_radians)) + width * np.abs(np.cos(angle_radians)))
# new_array = np.ones((new_height, new_width), dtype='uint8') * 255  

# for s_i in range(height):
#     for s_j in range(width):
#         s_point = np.array([s_i, s_j])
#         scale_point = rotation_matrix.dot(s_point)
#         i, j = scale_point[0], scale_point[1]
#         i = int(i)
#         j = int(j)
#         new_array[i, j] = array[s_i, s_j]  

# new_image = Image.fromarray(new_array)
# new_image.show()



# rotation_do = rotation('a2.jpg',45)



#--------------------------------------------------------------------------

# img = Image.open('flower.png')   
# img_size = img.size
# width, height = img_size
# pixels = list(img.getdata())
# array = np.array(pixels,dtype='uint8')[:,0].reshape(height, width)  # این فقط برای عکس های سیاه و سفید توی فرمت آرجی بی ای هست که خودش به صورت باینری ذخیره میکنه 
# # print(array)


##    اروژن 
# kernel_ones  = np.ones(shape=(3,3), dtype='uint8')
# new_array = np.zeros(shape=(height,width), dtype='uint8')

# for i in range(1, height-1):
#     for j in range(1 , width- 1):
#         window = array[i - 1:i + 2, j - 1:j + 2]
#         if (window * kernel_ones).all() :
#             new_array[i,j] = 255
#         else :
#             new_array[i,j] = 0


# new_image = Image.fromarray(new_array)
# new_image.show()
    
####

##    دایرکشن   
# kernel_ones  = np.ones(shape=(3,3), dtype='uint8')
# new_array = np.zeros(shape=(height,width), dtype='uint8')

# for i in range(1, height-1):
#     for j in range(1 , width- 1):
#         window = array[i - 1:i + 2, j - 1:j + 2]
#         if (window * kernel_ones).all() :
#             new_array[i,j] = 255
#         else :
#             new_array[i,j] = 0


# new_image = Image.fromarray(new_array)
# new_image.show()


#------------------------------------------    for noise in new_image 
# 644017  # image noise
# 134996  # FLOWER

####
# array, height, width = convert_to_array(image='img.png')
# kernel_ones  = np.ones(shape=(2,2), dtype='uint8')
# new_array = np.zeros(shape=(height, width), dtype='uint8')
# #
# for i in range(1, height - 1):  # از 1 شروع می‌کنیم تا لبه‌های تصویر را تغییر ندهیم
#     for j in range(1, width - 1):
#         window = array[i - 1:i + 2, j - 1:j + 2]
#         mean_value = np.mean(window)
#         threshold = 180  # img.png 180    # flower.pn  250    # img2.jpg 50
#         # اگر میانگین از آستانه بیشتر بود، آن را سفید (255) تنظیم کن، در غیر این صورت سیاه (0)
#         if mean_value > threshold:
#             new_array[i, j] = 255
#         else:
#             new_array[i, j] = 0

# new_image = Image.fromarray(new_array)
# new_image.show()

# Remove_noise(image='img2.jpg',threshold=50)
    
#############





# # Load the image using OpenCV
# image_path = 'img_text.PNG'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Apply adaptive thresholding to binarize the image
# # This automatically detects local variations in brightness to separate text from background
# binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 16)

# # Apply morphological transformations to clean up the image
# kernel = np.ones((2,2), np.uint8)

# # Erosion followed by dilation (opening) to remove small black regions (noises)
# morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# # Convert the result back to an image and save
# result_image = Image.fromarray(morph_image)
# result_image.show()









#----------------------------






array, height, width = convert_to_array(image='i2.png') 

# kernel = np.ones(shape=(5,5) , dtype='uint8') * 1/25
##   1
# kernel = np.array([
#     [ 0  , 0 , 0   ],
#     [ 1/9, 1/9 ,1/9 ],
#     [ 2/9, 2/9 ,2/9 ] 
# ])
##   2
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype='float32') * (1/9)


##  3
# kernel = np.array([
#     [-1,  0,  1],
#     [-1,  0,  1],
#     [-1,  0,  1]
# ], dtype='float32') / 3
## 4
# kernel = np.array([
#     [-1, -1, -1],
#     [ 0,  1,  0],
#     [ 1,  1,  1]
# ], dtype='float32') * 2

# new_array = np.zeros(shape=(height,width), dtype='uint8')

# for i in range(2, height-3):
#     for j in range(2 , width - 3):
#         # window = array[i - 2:i + 3, j - 2:j + 3]
#         window = array[i - 1:i + 2, j - 1:j + 2]
#         window_sum = np.sum(window * kernel) 
#         new_array[i, j] = window_sum


# new_image = Image.fromarray(new_array)
# new_image.show()

###########  

kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype='float32') * (1/9)

kernel_2 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype='float32') * (1/9)

new_array = np.zeros(shape=(height,width), dtype='uint8')

for i in range(2, height-3):
    for j in range(2 , width - 3):
        # window = array[i - 2:i + 3, j - 2:j + 3]
        window = array[i - 1:i + 2, j - 1:j + 2]
        window_sum_1 = np.sum(window * kernel) 
        window_sum_2 = np.sum(window * kernel_2) 
        
        new_array[i, j] = window_sum_1 + window_sum_2


new_image = Image.fromarray(new_array)
new_image.show()







##################      exponential 


##################     gaussian function





