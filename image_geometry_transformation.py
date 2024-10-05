import numpy as np 
from PIL import Image , ImageOps
import math

# image geometry transformation  


# scale / size transformation

# reflection

# rotation

#  shear 

#----------------------------- scale   تبدیلات هندسی 

img = Image.open('a2.jpg')   # 815 733
img_size = img.size
width, height = img_size
im_gray = ImageOps.grayscale(img)
pixels = list(im_gray.getdata())
array = np.array(pixels,dtype='uint8').reshape(height, width)
scale_array = np.array([ [2, 0],
                        [0, 2]])
new_array = np.ones((2 *  height , 2 * width ) , dtype='uint8') * 255   # ساختن یه آرایه خالی و ضرب در 255 کردن چون میخوایم بک گراند سفید باشه 

#######################################  روش معمولی 
# for i in range(height):
#     for j in range(width):
#         point= np.array([i,j])
#         scale_point = scale_array.dot(point)
#         s_i , s_j = scale_point[0], scale_point[1]
#         new_array[s_i,s_j] = array[i,j]

#######################################
scale = np.linalg.inv(scale_array)
for s_i in range(height * 2):
    for s_j in range(width * 2):
        s_point = np.array([s_i, s_j])
        scale_point = scale.dot(s_point)
        i, j = scale_point[0], scale_point[1]
        i = int(math.floor(i))
        j = int(math.floor(j))
        new_array[s_i, s_j] = array[i, j]

new_image = Image.fromarray(new_array)
print(new_array.size)


# from PIL import Image
# import numpy as np

# # ۱. باز کردن فایل JPEG به عنوان یک تصویر
# with Image.open('a1.jpg') as img:
#     # تبدیل تصویر به آرایه NumPy
#     image_np = np.array(img)

# # ۲. دوبابر کردن پیکسل‌ها
# # مطمئن شوید که مقادیر پیکسل‌ها بین ۰ تا ۲۵۵ باقی بمانند
# doubled_image_np = np.clip(2 * image_np, 0, 255).astype(np.uint8)

# # ۳. تبدیل دوباره آرایه به تصویر
# doubled_image = Image.fromarray(doubled_image_np)

# # ۴. ذخیره تصویر دوبابر شده
# doubled_image.save('doubled_image.jpg')
