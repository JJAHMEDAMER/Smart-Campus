import cv2
import pandas as pd

IMG_PATH = 'test_images//1.jpg'
img = cv2.imread(IMG_PATH)

'''
Image Have three channels red green and Blue but we only care about the intensity
and the value af brightness so, we can turn the image to gray scale to get a single value 
that represent the intensity of each pixel
'''
img_int = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


'''
lets use pandas data frame to display our image as an array of values, and make sure that
the numeric representation is int 32 to avoid overflow when later we do arythmic operations 
the default value is int8 as the pixel values only goes up to 255

this is a conscious decision by cv2 lib but we will need a larger range

2^8 = 256
0 -> 255

2^32 = 4_294_967_296
0 -> 4_294_967_295
'''
img_int_df = pd.DataFrame(img_int).astype('int32')
print(img_int_df)


'''
If we take a look at the shape of the 3 channel img that is represented as an 
array of three dimension vs the shape of the intensity image we can see that 
the image has been flattened to one dimension
'''
print(img.shape)  # Original image
print(img_int_df.shape)  # Gray scale image


'''

'''

print(img_int_df.iloc[:5, :5])

sub_array = img_int_df.iloc[:5, :5]

for i in sub_array:
    if i == 0:
        continue
    sub_array[i] = sub_array[i] + sub_array[i-1]  # Columns
    sub_array.iloc[i] = sub_array.iloc[i] + sub_array.iloc[i-1]  # Rows
    
print(sub_array)

print((sub_array.dtypes))

