from Function_Ai import *



##################     gaussian function

# kernel = np.zeros((3,3) , dtype=np.float32 )       #  برای این فلوت میکنیم که عدد کامل رو بگیریم زیرا اینتیجر بدیم میاد روندش میکنه 
# sigma = 1

# for x in range(-1,2):
#     for y in range(-1,2):
#         gauss_value = (1 / (2 * np.pi * sigma ** 2 )) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#         kernel[x+1 , y+1] = gauss_value 

# print(kernel)


def gaussian_function (size , sigma) :
    kernel = np.zeros((size,size) )       #  برای این فلوت میکنیم که عدد کامل رو بگیریم زیرا اینتیجر بدیم میاد روندش میکنه 
    h = size // 2
    for x in range(-h,h + 1):
        for y in range(-h,h + 1):
            gauss_value = (1 / (2 * np.pi * sigma ** 2 )) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[x+h , y+h] = gauss_value 
    return kernel
# ############################
# k = gaussian_function(5,1)

# # print(k.sum())

# array, height, width = convert_to_array(image='image/mous.png') 

# smoothed_array = np.zeros(shape=(height,width), dtype='uint8')


# for i in range(10, height-10):
#     for j in range(10 , width - 10):
#         window = array[i - 2:i + 3, j - 2:j + 3]
#         ij_smoothed = np.sum(window * k) 
#         smoothed_array[i, j] = ij_smoothed

# diff_array_x = np.zeros(shape=(height,width), dtype='uint8')
# diff_array_y = np.zeros(shape=(height,width), dtype='uint8')
# mask = [ -1 , 1 ]

# for i in range(10, height-10):
#     for j in range(10 , width - 10):
#         diff_x = smoothed_array[i+1 , j] - smoothed_array[i , j ]
#         diff_array_x [i,j] = diff_x

#         diff_y = smoothed_array[i , j+1 ] - smoothed_array[i , j ]
#         diff_array_y [i,j] = diff_x



# diff = (diff_array_x**2 + diff_array_y**2) **0.5   # بر آیند ایکس و وای 
# diff = diff > 1.43   # اگر از یک مقداری بیشتر بود ترو بده اگر هم نه فالس 

# new_image = Image.fromarray(diff)   # این تابع بر اساس ترو و فالس هم پیکسل گذاری میکنه ترو رو سفید میزاره و فالس رو مشکی
# new_image.show()



######################


# def filter_2d(image_array, mask):
#     height, width = image_array.shape
#     window_size = mask.shape[0]
#     w = window_size // 2
#     filtered_array = np.zeros_like(image_array, dtype='float32')
#     for i in range(w, height - w):
#         for j in range(w, width - w):
#             window = image_array[i - w:i + w + 1, j - w:j + w + 1]
#             filtered_array[i, j] = np.sum(window * mask)
#     return filtered_array



# def moshtagh (image_array ) :
#     height , width  = image_array.shape
#     diff_array_x = np.zeros(shape=(height,width), dtype='uint8')
#     diff_array_y = np.zeros(shape=(height,width), dtype='uint8')
#     for i in range(10, height-10):
#         for j in range(10 , width - 10):
#             diff_array_x = -i * np.exp(-(i ** 2 + j** 2) )
            
            
#     return filtered_array

# kernel = gaussian_function(5,1)
# image_array = convert_to_array_just_array(image='image/mous.png')
# image_filter = filter_2d(image_array=image_array , mask=kernel)
# new_image = Image.fromarray(image_filter)
# new_image.show()


##########################     تشخیص جسم با تابع گوسین 



# def gaussian(size, sigma):
#     kernel_x = np.zeros((size, size))
#     kernel_y = np.zeros((size, size))
#     h = size // 2
#     for x in range(-h, h + 1):
#         for y in range(-h, h + 1):
#             gauss_value = np.exp(-(x**2 + y**2) / (2 * sigma**2))
#             kernel_x[x + h, y + h] = -x * gauss_value
#             kernel_y[x + h, y + h] = -y * gauss_value
#     return kernel_x, kernel_y




# size = 5
# sigma = 1

# kernel = gaussian_function(size, sigma)
# image_array = convert_to_array_just_array('image/mous.png')
# smoothed_image = filter_2d(image_array, kernel)

# kernel_x, kernel_y = gaussian(size, sigma)
# diff_array_x = filter_2d(smoothed_image, kernel_x)
# diff_array_y = filter_2d(smoothed_image, kernel_y)

# diff = np.sqrt(diff_array_x**2 + diff_array_y**2)
# # new_image = rotation(image=diff , angle=90)
# new_image = Image.fromarray((diff / diff.max() * 255).astype('uint8'))
# new_image.show()







# diff = (diff_array_x**2 + diff_array_y**2) **0.5   # بر آیند ایکس و وای 
# diff = diff > 180
#    # اگر از یک مقداری بیشتر بود ترو بده اگر هم نه فالس 

# new_image = Image.fromarray(diff)   # این تابع بر اساس ترو و فالس هم پیکسل گذاری میکنه ترو رو سفید میزاره و فالس رو مشکی
# new_image.show()
#########################


