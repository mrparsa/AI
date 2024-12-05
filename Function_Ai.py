import numpy as np 
from PIL import Image , ImageOps

#--------------------- def covert
def convert_to_array(image):
    img = Image.open(image)   # 815 733
    img_size = img.size
    width, height = img_size
    im_gray = ImageOps.grayscale(img)
    pixels = list(im_gray.getdata())
    array = np.array(pixels,dtype='uint8').reshape(height, width)
    return array , height , width

def convert_to_array_just_array(image):
    img = Image.open(image)   # 815 733
    img_size = img.size
    width, height = img_size
    im_gray = ImageOps.grayscale(img)
    pixels = list(im_gray.getdata())
    array = np.array(pixels,dtype='uint8').reshape(height, width)
    return array 
#------------------------   فانکشن دوران 
def rotation (image,angle) :

# تبدیل زاویه به رادیان
    if isinstance(image, Image.Image) :
        array =  convert_to_array_just_array(image=image)
        height , width = array.shape
    else :
        array = image
        height , width = image.shape

    angle_radians = np.deg2rad(angle)

    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],  
        [np.sin(angle_radians), np.cos(angle_radians)]    
    ])

    new_height = int(height * np.abs(np.cos(angle_radians)) + width * np.abs(np.sin(angle_radians)))
    new_width = int(height * np.abs(np.sin(angle_radians)) + width * np.abs(np.cos(angle_radians)))
    new_array = np.ones((new_height, new_width), dtype='uint8') * 255  # ایجاد پس‌زمینه سفید

    for s_i in range(height):
        for s_j in range(width):
            s_point = np.array([s_i, s_j])
            scale_point = rotation_matrix.dot(s_point)
            i, j = scale_point[0], scale_point[1]
            i = int(i)
            j = int(j)
            new_array[i, j] = array[s_i, s_j]  # مقدار پیکسل را منتقل کنید

    new_image = Image.fromarray(new_array)
    new_image.show()


#--------------------------------------   def for noise of image 
def Remove_noise ( image , threshold )   :  
    array, height, width = convert_to_array(image=image)
    # kernel_ones  = np.ones(shape=(2,2), dtype='uint8')
    new_array = np.zeros(shape=(height, width), dtype='uint8')
    for i in range(1, height - 1):  # از 1 شروع می‌کنیم تا لبه‌های تصویر را تغییر ندهیم
        for j in range(1, width - 1):
            window = array[i - 1:i + 2, j - 1:j + 2]
            mean_value = np.mean(window)
            threshold = threshold   # img.png 180    # flower.pn  250    # img2.jpg 50
            # اگر میانگین از آستانه بیشتر بود، آن را سفید (255) تنظیم کن، در غیر این صورت سیاه (0)
            if mean_value > threshold :
                new_array[i, j] = 255
            else:
                new_array[i, j] = 0

    new_image = Image.fromarray(new_array)
    new_image.show()




###    gaussian function

def gaussian_function (size , sigma) :
    kernel = np.zeros((size,size) )       #  برای این فلوت میکنیم که عدد کامل رو بگیریم زیرا اینتیجر بدیم میاد روندش میکنه 
    h = size // 2
    for x in range(-h,h + 1):
        for y in range(-h,h + 1):
            gauss_value = (1 / (2 * np.pi * sigma ** 2 )) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[x+h , y+h] = gauss_value 
    return kernel



def filter_2d(image_array, mask):
    height, width = image_array.shape
    window_size = mask.shape[0]
    w = window_size // 2
    filtered_array = np.zeros_like(image_array, dtype='float32')
    for i in range(w, height - w):
        for j in range(w, width - w):
            window = image_array[i - w:i + w + 1, j - w:j + w + 1]
            filtered_array[i, j] = np.sum(window * mask)
    return filtered_array




def gaussian(size, sigma):
    kernel_x = np.zeros((size, size))
    kernel_y = np.zeros((size, size))
    h = size // 2
    for x in range(-h, h + 1):
        for y in range(-h, h + 1):
            gauss_value = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel_x[x + h, y + h] = -x * gauss_value
            kernel_y[x + h, y + h] = -y * gauss_value
    return kernel_x, kernel_y
