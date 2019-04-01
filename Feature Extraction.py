import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import glob
import os
import time

last_time = time.clock()
# ignore the information from hardware speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # read the inlier_train information
# file_path = 'E:\\python\\5002DM\\ITSC_20527755_MSCBDT5002_final\\Q1\\Q1_code\\Data_Q1\\inlier_train\\'
# f_names = glob.glob(file_path+'*.jpg')

# # read the outlier_train information
# file_path = 'Data_Q1/outlier_train/'
# f_names = glob.glob(file_path+'*.jpg')
#
# read the test information
file_path = 'Data_Q1/test/'
f_names = glob.glob(file_path+'*.jpg')

# load the model
model = ResNet50(weights='imagenet', include_top=False, pooling='max')

# load the image, predict the feature and put them in the list
Resnet_feature_list = []
for i in range(len(f_names)):
    images = image.load_img(f_names[i], target_size=(224, 224))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    resnet_feature = model.predict(x)
    resnet_feature_np = np.array(resnet_feature)
    Resnet_feature_list.append(resnet_feature_np.flatten())
    print('feature_extractor for no.%s image'%i)

Resnet_feature_list_np = np.array(Resnet_feature_list)

# save the numpy.array into the csv so that we can load them for clustering in the next stage
np.savetxt('test.csv', Resnet_feature_list_np, delimiter=',')
current_time = time.clock()
print('total time consuming for feature extractor: {}'.format(current_time-last_time))
