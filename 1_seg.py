# Manually add envs to the python path
import os
import sys

paths = [
        "./cython0",
        "./cython2",
        "./cython3"
        ]

for path in paths:
    sys.path.append(path)

import numpy as np
import pydicom

import SimpleITK as sitk
import tensorflow as tf
import vtk

from cython_voxel_layer import voxel_layer
from cython_extend_boundary import extend_boundary
from cython_reduce_boundary import reduce_boundary

from scipy import ndimage
from scipy.ndimage import zoom
from vtkmodules.util import numpy_support

#from paraview.simple import *
#from paraview import servermanager as sm

import nrrd
import time

from skimage.measure import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D, GlobalMaxPooling3D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#Extend boundaries
write_ct_data_bool=True
write_filtered_data_bool=True
series_number_bool=True
series_number=5

KLI_bool=True

seg_ext_bool=True
seg_red_bool=False
seg_ext_nr=2

#use external dicom
use_dicom=True
ext_dicom="ct_data.nrrd" #only needed if use_dicom = False
ext_RS=1 #only needed if use_dicom = False
ext_RI=0 #only needed if use_dicom = False
VS_X=0.326171875 #only needed if use_dicom = False
VS_Y=0.326171875 #only needed if use_dicom = False
VS_Z=0.7 #always needed !!!!!

#use external segmentation
use_ext_seg=False
ext_seg_name="segmentation_initial_plan.nrrd"

#Force outlet cutting, in case surface based on external segmentation is not open
force_out_cut=False
force_OUT_c=[69.84113836288452,96.89016723632812,7.257740599237669]
force_OUT_n=[0,0,-1]

#Normalized distance in each z-direction to predict inlets
#Normalized by x bound of nasal cavity (L_x): z_dist_nor=z_dist/L_x=nr_sl_z*voxel_size[2]/L_x
z_dist_nor=0.2

# Setup data directories
base_path = '.' #sys.argv[1]
data_dir = os.path.join(base_path, 'dicom_test')
output_dir = os.path.join(base_path, 'geometry_extraction')
# Create the output directory if necessary
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

voxel_size_file = "voxel_size.nrrd"
output_file = "CTraw.stl"
output_file_ext = "CTraw_ext.stl"
inlet_left_file = "inlet_left.nrrd"
inlet_right_file = "inlet_right.nrrd"
segmentation_file = "segmentation.nrrd"
dice_file = "dice.nrrd"
seg_dice_file = "seg_dice.nrrd"
ct_file = "ct_data.nrrd"
filtered_ct_file = "ct_data_filtered.nrrd"
span_file = "span.nrrd"
seg_ext_file = "seg_ext.nrrd"

# Define functions necessary for the segmentation
# Function for 2D convolutional block inside network architecture
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # First layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Second layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# Function for 2D network architecture
def get_net_2D(input_img, out_layer, n_filters, dropout, batchnorm=True):
    # Contracting path
    c1 = conv2d_block(
        input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling2D((2, 2))(c1)
    #p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    #p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    #p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(c5)
    #u6 = concatenate([u6, c4])
    sc6 = add([u6,c4])
    #u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(sc6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(c6)
    #u7 = concatenate([u7, c3])
    sc7 = add([u7,c3])
    #u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(sc7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(c7)
    #u8 = concatenate([u8, c2])
    sc8 = add([u8,c2])
    #u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(sc8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(c8)
    #u9 = concatenate([u9, c1], axis=3)
    sc9 = add([u9,c1])
    sc9 = Dropout(dropout)(sc9)
    c9 = conv2d_block(sc9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(out_layer, (1, 1), activation="sigmoid")(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# Function for 3D convolutional block inside network architecture
def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    # first layer
    x_1 = Conv3D(filters=n_filters,
                 kernel_size=(kernel_size, kernel_size, kernel_size),
                 kernel_initializer="he_normal",
                 padding="same")(input_tensor)
    if batchnorm:
        x_2 = BatchNormalization()(x_1)
    x_2 = Activation("relu")(x_2)

    # second layer
    x_2 = Conv3D(filters=n_filters,
                 kernel_size=(kernel_size, kernel_size, kernel_size),
                 kernel_initializer="he_normal",
                 padding="same")(x_2)
    if batchnorm:
        x_2 = BatchNormalization()(x_2)
    x_2 = Activation("relu")(x_2)

    return x_2

# Function for 3D network architecture
def get_net_3D(input_img, n_filters, dropout, batchnorm=True):

    # contracting path
    c1 = conv3d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling3D((2, 2, 2)) (c1)

    c2 = conv3d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling3D((2, 2, 2)) (c2)

    c3 = conv3d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling3D((2, 2, 2)) (c3)

    c4 = conv3d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling3D(pool_size=(2, 2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv3d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv3DTranspose(n_filters*8, (3, 3, 3), strides=(2, 2, 2), padding='same') (c5)
    sc6 = add([u6, c4])
    c6 = conv3d_block(sc6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv3DTranspose(n_filters*4, (3, 3, 3), strides=(2, 2, 2), padding='same') (c6)
    sc7 = add([u7,c3])
    c7 = conv3d_block(sc7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv3DTranspose(n_filters*2, (3, 3, 3), strides=(2, 2, 2), padding='same') (c7)
    sc8 = add([u8,c2])
    c8 = conv3d_block(sc8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv3DTranspose(n_filters*1, (3, 3, 3), strides=(2, 2, 2), padding='same') (c8)
    sc9 = add([u9,c1])
    sc9 = Dropout(dropout)(sc9)
    c9 = conv3d_block(sc9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# Function for "keep largest island"
def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    # the 0 label is by default background so take the rest
    list_seg = list(zip(unique, counts))[1:]
    largest = max(list_seg, key=lambda x:x[1])[0]
    labels_max = (labels == largest).astype(int)
    return labels_max

# Start of segmentaion
print("#"*30)
print("Segmentation")
print("#"*30)

start_step_i=time.time()

if use_dicom==True:

   # Load the DICOM files
   print("Loading DICOM files...", flush=True)
   files = []
   for file in os.listdir(data_dir):
       file_path = os.path.join(data_dir, file)
       #print("file path: ", ""+str(file_path)+"", flush=True)
       if os.path.isfile(file_path):
           try:
               ds=pydicom.filereader.dcmread(file_path)
               if series_number_bool==True:
                   #print("series number: ", ds.SeriesNumber, flush=True)
                   if ds.SeriesNumber==series_number: #and ds.SeriesTime=='085446':
                       files.append(pydicom.read_file(file_path))
                       #files.append(dicom.read_file(""+str(file_path)+""))
               else:
                  files.append(pydicom.read_file(file_path))
           except pydicom.errors.InvalidDicomError:
               pass
   print("Number of files: {}".format(len(files)), flush=True)

   # Skip files with no slice location (eg scout views)
   slices = []
   skipcount = 0
   for f in files:
       if hasattr(f, "SliceLocation") or hasattr(f, "SliceThickness"):
           slices.append(f)
       else:
           skipcount += 1
   del files
   print("Skipped, no SliceLocation: {}".format(skipcount), flush=True)

   # Ensure slices are in the correct order
   slices = sorted(slices, key=lambda s: s.SliceLocation)
   # There are two properties that describe slices in z direction:
   # "Slice thickness" and "Spacing between slices”. 
   # We need “Spacing between slices” for creating the .stl later.
   #print("Spacing between slices: %f" % slices[0].SpacingBetweenSlices, flush=True)
   print("Pixel spacing (row, col): ({}, {})".format(
       slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]), flush=True)

   #Write pixel spacing and spacing between slices
   voxel_size=np.zeros((3))
   voxel_size[0]=float(slices[0].PixelSpacing[0])
   voxel_size[1]=float(slices[0].PixelSpacing[1])
   #voxel_size[2]=float(slices[0].SliceThickness)
   voxel_size[2]=float(VS_Z)
   print("voxel_size[2]: ", voxel_size[2], flush=True)

   # Create 3D array
   img_shape = list(slices[0].pixel_array.shape)
   img_shape.append(len(slices))
   img3d = np.empty(img_shape)

   # Fill 3D array with the images from the files
   for i, s in enumerate(slices):
       img3d[:, :, i] = s.pixel_array # pixel_array = img2d

   # Reshape 3D array and recalculate voxel size if x,y dimensions are not 512x512
   #X axis
   if slices[0].pixel_array.shape[0] != 512:
      x_res=np.round((512/slices[0].pixel_array.shape[0]),decimals=3)
      img3d = zoom(img3d, (x_res, 1, 1))

      voxel_size[0]=(voxel_size[0]/slices[0].pixel_array.shape[0])*512

   #Y axis
   if slices[0].pixel_array.shape[1] != 512:
      y_res=np.round((512/slices[0].pixel_array.shape[1]),decimals=3)
      img3d = zoom(img3d, (1, y_res, 1))

      voxel_size[1]=(voxel_size[1]/slices[0].pixel_array.shape[1])*512

   #Swap y-axis, to match with CT data
   #img3d_rev = np.empty((img3d.shape[0],img3d.shape[1],img3d.shape[2]))
   #for i in range(img3d.shape[1]):
   #    img3d_rev[:,i,:]=img3d[:,(-i+(img3d.shape[1]-1)),:]

   #img3d=img3d_rev

   # If necessary, convert from CT values to HU units
   RS=slices[0].RescaleSlope
   RI=slices[0].RescaleIntercept
   print("\nRS: ",RS,flush=True)
   print("\nRI: ",RI,flush=True)
   

else:
   
   #read external dicom 
   img3d, header = nrrd.read(os.path.join(output_dir, ext_dicom))
   #Write pixel spacing and spacing between slices
   voxel_size=np.zeros((3))
   voxel_size[0]=float(VS_X)
   voxel_size[1]=float(VS_Y)
   #voxel_size[2]=float(slices[0].SliceThickness)
   voxel_size[2]=float(VS_Z)


   RS=ext_RS
   RI=ext_RI
   
nrrd.write(os.path.join(output_dir, voxel_size_file), voxel_size )

img3d = img3d * RS + RI

print("\nSetting up network...", flush=True)
# Kernel for convolution filter
k = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
# Initialisation of output file
CONV_and_GAD_filter = np.empty((img3d.shape[0], img3d.shape[1], img3d.shape[2]))

for i in range(img3d.shape[0]):
    slice_conv_filter = ndimage.convolve(img3d[i,:,:], k, mode='constant', cval=0)
    slice_GAD = sitk.GetImageFromArray(slice_conv_filter)
    slice_GAD = sitk.Cast(slice_GAD, sitk.sitkFloat32)
    GAD = sitk.GradientAnisotropicDiffusionImageFilter()
    GAD.SetNumberOfIterations(8)
    GAD.SetTimeStep(0.0625)
    GAD.SetConductanceParameter(5.0)
    filtered_slice = GAD.Execute(slice_GAD)
    CONV_and_GAD_filter[i,:,:] = sitk.GetArrayFromImage(filtered_slice)


#Write out raw CT data as array
if write_ct_data_bool==True:
   nrrd.write(os.path.join(output_dir, ct_file), img3d)
#Write out filtered CT data as array
if write_filtered_data_bool==True:
   nrrd.write(os.path.join(output_dir, filtered_ct_file), CONV_and_GAD_filter)

#######################
#CNN-A
#######################

# Data for normalization 
X_std_1 = 1551.049032612434
X_mean_1 = -1252.5419529469366
# Define input
X = CONV_and_GAD_filter
# Normalize input
X_norm_1 = (X - X_mean_1) / X_std_1
# Reshape test data to match with code later
X_norm_res_1 = np.empty((X_norm_1.shape[2], 512, 512, 1))

for i in range(X_norm_1.shape[2]):
    X_norm_res_1[i,:,:,0] = X_norm_1[:,:,i]

#Initialize network for segmentation
print("Initializing network for segmentation...", flush=True)
input_img_1 = Input((512, 512, 1), name='img')
model_1 = get_net_2D(input_img_1, out_layer=1, n_filters=32, dropout=0, batchnorm=True)
# model.summary()

# Load weights and biases of trained network for segmentation
model_1.load_weights('./seg_A.h5')

print("Starting segmentation...", flush=True)
#Start segmentation
segmentation = model_1.predict(X_norm_res_1, verbose=1)
#Set pixel quantities to binary values 0 or 1 around threshold 0.5
segmentation[segmentation > 0.5] = 1.0
segmentation[segmentation <= 0.5] = 0
#Remove 4th dimension of segmentation
segmentation_res=np.zeros((X.shape[2],X.shape[0],X.shape[1]))
for i in range(X.shape[2]):
    segmentation_res[i]=segmentation[i,:,:,0]
#Transpose segmentation to x,y,z coordinates
segmentation_tra = np.transpose(segmentation_res, (1, 2, 0))

# Keep largest island
if KLI_bool==True:
   segmentation_tra = getLargestCC(segmentation_tra)

#Save as .nrrd file
nrrd.write(os.path.join(output_dir, segmentation_file), segmentation_tra)

#######################
#CNN-B
#######################

#Identify coordinates of first segmented voxel in x-direction
#Based on that, nostril dice is refined and thin inlet layers are predicted 
#+/- a certain number of slices in z-direction around that point

#Loop through the first 1/3 of the segmentation in x-direction
#to find the spanwidth in z-direction between frontal sinuses and nostrils

span_up=0
span_down=segmentation_tra.shape[2]

#Loop through segmentation
for i in range(int(segmentation_tra.shape[0]/3)):
   for j in range(segmentation_tra.shape[1]):
      for k in range(segmentation_tra.shape[2]):

         #Update span_up and span_down
         if segmentation_tra[i,j,k]==1 and k>span_up:

            span_up=k
         
         if segmentation_tra[i,j,k]==1 and k<span_down:

            span_down=k

print("span_up: ", span_up)
print("span_down: ", span_down)

#Save as .nrrd file
span=np.zeros((2))
span[0]=span_up
span[1]=span_down
nrrd.write(os.path.join(output_dir, span_file), span)

#Loop through segmentation from 0 to the half spanwidth in z-direction
#to find minimum x coordinate of nostrils
flag=True

for i in range(segmentation_tra.shape[0]):
   for j in range(segmentation_tra.shape[1]):
      for k in range(int(span_down+(span_up-span_down)/2)):

         #When finding the first segmented voxel, break all loops and store the z position
         if segmentation_tra[i,j,k]==1:

            x_min_coord=[i,j,k]

            flag=False
            break

      if flag==False:

         break

   if flag==False:

      break

print("x_min_coord: ",x_min_coord,flush=True)

#Predict dice around nostrils with 3D CNN
#Size of dice
x_edge=80
y_edge=96
z_edge=48

#Steps from x_min_coord
x_back=8
x_front=72
y_back=48
y_front=48
z_back=28
z_front=20

#Dice from filtered CT data: dice_X
#Check if dice exceeds z-bounds of CT data
if x_min_coord[2]-z_back<0:

   dice_X = X[int(x_min_coord[0]-x_back):int(x_min_coord[0]+x_front),
              int(x_min_coord[1]-y_back):int(x_min_coord[1]+y_front),
              0:z_edge]

else:
    
   dice_X = X[int(x_min_coord[0]-x_back):int(x_min_coord[0]+x_front),
              int(x_min_coord[1]-y_back):int(x_min_coord[1]+y_front),
              int(x_min_coord[2]-z_back):int(x_min_coord[2]+z_front)]

#Data for normalization
X_std_2=1301.7722854480435
X_mean_2=-890.3835213936984

#Normalize input
X_norm_2=(dice_X-X_mean_2)/X_std_2

# Reshape test data to match with code later
X_norm_res_2=np.zeros((1,X_norm_2.shape[0],X_norm_2.shape[1],X_norm_2.shape[2],1))

X_norm_res_2[0,:,:,:,0] = X_norm_2[:,:,:]

#Initialize network for dice
print("Initializing network for prediction of dice around nostrils...", flush=True)
#Input
input_img_2 = Input((x_edge, y_edge, z_edge, 1), name='img')

#Initialize network
model_2 = get_net_3D(input_img_2, n_filters=32, dropout=0, batchnorm=True)

# Load weights and biases of trained network for segmentation
model_2.load_weights('./seg_B.h5')

print("Starting prediction of dice around nostrils...", flush=True)
#Start prediction
dice = model_2.predict(X_norm_res_2, verbose=1)

#Set pixel quantities to binary values 0 or 1 around threshold 0.5
dice[dice > 0.5] = 1.0
dice[dice <= 0.5] = 0

#Remove 1st and 4th dimension of dice
dice_res=np.zeros((dice.shape[1],dice.shape[2],dice.shape[3]))

dice_res=dice[0,:,:,:,0]

#Save as .nrrd file
nrrd.write(os.path.join(output_dir, dice_file), dice_res)

#Fill segmentation with predicted dice
if x_min_coord[2]-z_back<0:
   
    segmentation_tra[int(x_min_coord[0]-x_back):int(x_min_coord[0]+x_front),
                     int(x_min_coord[1]-y_back):int(x_min_coord[1]+y_front),
                     0:z_edge] = dice_res

else:
    
    segmentation_tra[int(x_min_coord[0]-x_back):int(x_min_coord[0]+x_front),
                     int(x_min_coord[1]-y_back):int(x_min_coord[1]+y_front),
                     int(x_min_coord[2]-z_back):int(x_min_coord[2]+z_front)] = dice_res

# Keep largest island
segmentation_tra = getLargestCC(segmentation_tra)

#Save as .nrrd file
nrrd.write(os.path.join(output_dir, seg_dice_file), segmentation_tra)

end_step_i=time.time()
print('step (i)')
print('{:5.3f}s'.format(end_step_i-start_step_i))

#######################
#CNN-C
#######################

start_step_iii_1=time.time()

#Identify last segmented voxel in x-direction,
#to calculate the x bound of the nasal cavity
flag=True

#Loop through segmentation
for i in range(segmentation_tra.shape[0]-1,0,-1):
   for j in range(segmentation_tra.shape[1]):
      for k in range(segmentation_tra.shape[2]):

         #When finding the first segmented voxel, break all loops and store the z position
         if segmentation_tra[i,j,k]==1:

            flag=False

            x_max_coord=[i,j,k]

            break

      if flag==False:

         break

   if flag==False:

      break

#x bound of nasal cavity
L_x=(x_max_coord[0]-x_min_coord[0])*voxel_size[0]

#Nr of slices in each z-direction to predict thin inlet layers
nr_sl_z=int((z_dist_nor/voxel_size[2])*L_x)

#Initialize network for left inlet
print("Initializing network for inlets...", flush=True)
input_img_3 = Input((512, 512, 2), name='img')
model_3 = get_net_2D(input_img_3, out_layer=2, n_filters=32, dropout=0, batchnorm=True)
# model.summary()

# Load weights and biases of trained network for left inlet
model_3.load_weights('./seg_C.h5')

#Data for normalization
X_std_3=1551.7995144825823
X_mean_3=-1248.3770373938012

#Reshape and normalize data
X_norm_res_3 = np.empty((2*nr_sl_z, 512, 512, 2))

print("nr_sl_z: ",nr_sl_z,flush=True)

#Check if x_min_coord is to close to the min_z of the CT data
if (x_min_coord[2]-nr_sl_z)<0:

    for i in range(2*nr_sl_z):
       
       X_norm_res_3[i,:,:,0] = (X[:,:,i] - X_mean_3) / X_std_3
       X_norm_res_3[i,:,:,1] = segmentation_tra[:,:,i]

else:

    for i in range(x_min_coord[2]-nr_sl_z,x_min_coord[2]+nr_sl_z):

       X_norm_res_3[i+nr_sl_z-x_min_coord[2],:,:,0] = (X[:,:,i] - X_mean_3) / X_std_3
       X_norm_res_3[i+nr_sl_z-x_min_coord[2],:,:,1] = segmentation_tra[:,:,i]

print("Starting prediction of inlets...", flush=True)
#Start prediction of left inlet
inlets = model_3.predict(X_norm_res_3, verbose=1)
#Set pixel quantities to binary values 0 or 1 around threshold 0.5
inlets[inlets > 0.5] = 1.0
inlets[inlets <= 0.5] = 0

#Remove 4th dimension of inlets
inlet_left=np.zeros((2*nr_sl_z,X.shape[0],X.shape[1]))
inlet_right=np.zeros((2*nr_sl_z,X.shape[0],X.shape[1]))

for i in range(2*nr_sl_z):
    inlet_left[i]=inlets[i,:,:,0]
    inlet_right[i]=inlets[i,:,:,1]

#Transpose to x,y,z coordinates
inlet_left_tra = np.transpose(inlet_left, (1, 2, 0))
inlet_right_tra = np.transpose(inlet_right, (1, 2, 0))

#Fill complete array
if (x_min_coord[2]-nr_sl_z)<0:

    inlet_left_full=np.zeros((X.shape[0],X.shape[1],X.shape[2]))
    inlet_left_full[:,:,0:2*nr_sl_z]=inlet_left_tra
    
    inlet_right_full=np.zeros((X.shape[0],X.shape[1],X.shape[2]))
    inlet_right_full[:,:,0:2*nr_sl_z]=inlet_right_tra

else:

    inlet_left_full=np.zeros((X.shape[0],X.shape[1],X.shape[2]))
    inlet_left_full[:,:,x_min_coord[2]-nr_sl_z:x_min_coord[2]+nr_sl_z]=inlet_left_tra

    inlet_right_full=np.zeros((X.shape[0],X.shape[1],X.shape[2]))
    inlet_right_full[:,:,x_min_coord[2]-nr_sl_z:x_min_coord[2]+nr_sl_z]=inlet_right_tra

# Keep largest island
inlet_left_full = getLargestCC(inlet_left_full)
inlet_right_full = getLargestCC(inlet_right_full)

#Save as .nrrd file
nrrd.write(os.path.join(output_dir, inlet_left_file), inlet_left_full )
nrrd.write(os.path.join(output_dir, inlet_right_file), inlet_right_full )

end_step_iii_1=time.time()
print('step (iii) 1')
print('{:5.3f}s'.format(end_step_iii_1-start_step_iii_1))

start_step_ii=time.time()
# print("\nFilling holes and create new image data...", flush=True)
print("\nCreating new image data......", flush=True)
#gt_binary = ndimage.binary_fill_holes(segmentation_tra).astype(int)

#Use external segmentation
if use_ext_seg==True:

    segmentation_tra, header = nrrd.read(os.path.join(output_dir, ext_seg_name))


del segmentation
del segmentation_res
del X
del X_norm_1
del X_norm_res_1
del X_norm_2
del X_norm_res_2
del X_norm_res_3
del inlet_left
del inlet_left_tra
del inlet_left_full
del inlet_right
del inlet_right_tra
del inlet_right_full

# Initialize 2 voxel layer array
c = (segmentation_tra * -100000) + 50000
# Convert datatype to match with Cython code
c_input = c.astype(float)
a_input = CONV_and_GAD_filter.astype(float)
b_input = segmentation_tra.astype(float)
#Generate 2 voxel layer array
c_out = voxel_layer(b_input, a_input, c_input)

# Create a new vtk ImageData that contains the previously generated array.
# Here, it is important to use pixel dimensions and pixel spacing 
# from the original CT image (!).
newImageData = vtk.vtkImageData()
newImageData.SetDimensions(c.shape[0],c.shape[1],c.shape[2])
newImageData.SetSpacing(
    voxel_size
    #(float(slices[0].PixelSpacing[0])*0.001, 
    # float(slices[0].PixelSpacing[1])*0.001, 
    # float(slices[0].SliceThickness)*0.001)
)
newImageData.origin = (0, 0, 0)
newImageData.AllocateScalars(vtk.VTK_INT, 1)

for k in range(c.shape[2]):
    for j in range(c.shape[1]):
        for i in range(c.shape[0]):
            newImageData.SetScalarComponentFromDouble(i, j, k, 0, c_out[i,j,k])

# Apply marching cubes algorithm
marching = vtk.vtkMarchingCubes()
marching.SetInputData(newImageData)
marching.SetValue(0, -550)
marching.Update()
# Smooth the surface with the windowed sinc filter
lineStrip = vtk.vtkStripper()
lineStrip.SetInputConnection(marching.GetOutputPort())
smoother = vtk.vtkWindowedSincPolyDataFilter()
smoother.SetInputConnection(lineStrip.GetOutputPort())
smoother.SetNumberOfIterations(200) #100
smoother.SetPassBand(0.1) #0.2

print("\nWriting results to file...", flush=True)
# Write to .STL file
writer = vtk.vtkSTLWriter()
writer.SetInputConnection(smoother.GetOutputPort())
writer.SetFileTypeToBinary()
writer.SetFileName(os.path.join(output_dir, output_file))
writer.Write()

if force_out_cut==True:

    cT_stl = STLReader(FileNames=[os.path.join(output_dir, output_file)])
    cT_stl.UpdatePipeline()

    clip = Clip(Input=cT_stl)
    clip.ClipType = "Plane"
    clip.Scalars = ["CELLS", "STLSolidLabeling"]

    clip.ClipType.Origin = [force_OUT_c[0], force_OUT_c[1], force_OUT_c[2]]
    clip.ClipType.Normal = [force_OUT_n[0], force_OUT_n[1], force_OUT_n[2]]
    clip.Invert = 1

    extractSurface = ExtractSurface(Input=clip)
    triangulate = Triangulate(Input=extractSurface)

    SaveData(os.path.join(output_dir, output_file), proxy=triangulate, FileType="Ascii")

print("Segmentation done.\n")

end_step_ii=time.time()
print('step (ii)')
print('{:5.3f}s'.format(end_step_ii-start_step_ii))

#Boundary extension
if seg_ext_bool==True or seg_red_bool==True:
      
   SEG = segmentation_tra.astype(float)
   SEG_NEW = segmentation_tra.astype(float)
   
   #loop over number of extensions
   for i in range(seg_ext_nr):

      SEG = SEG.astype(float)
      SEG_NEW = SEG_NEW.astype(float)

      if seg_ext_bool==True:
         SEG=extend_boundary(SEG,SEG_NEW)
      elif seg_red_bool==True:
         SEG=reduce_boundary(SEG,SEG_NEW)
         
      SEG_NEW=SEG

   # Keep largest island
   if KLI_bool==True:
      SEG = getLargestCC(SEG)

   #write extended segmentation
   nrrd.write(os.path.join(output_dir, seg_ext_file), SEG)

   # Initialize 2 voxel layer array
   c = (SEG * -100000) + 50000

   # Convert datatype to match with Cython code
   c_input = c.astype(float)
   a_input = CONV_and_GAD_filter.astype(float)
   b_input = SEG.astype(float)
   #Generate 2 voxel layer array
   c_out = voxel_layer(b_input, a_input, c_input)

   # Create a new vtk ImageData that contains the previously generated array.
   # Here, it is important to use pixel dimensions and pixel spacing
   # from the original CT image (!).
   newImageData = vtk.vtkImageData()
   newImageData.SetDimensions(c.shape[0],c.shape[1],c.shape[2])
   newImageData.SetSpacing(
       voxel_size
       #(float(slices[0].PixelSpacing[0])*0.001,
       # float(slices[0].PixelSpacing[1])*0.001,
       # float(slices[0].SliceThickness)*0.001)
   )
   newImageData.origin = (0, 0, 0)
   newImageData.AllocateScalars(vtk.VTK_INT, 1)

   for k in range(c.shape[2]):
       for j in range(c.shape[1]):
           for i in range(c.shape[0]):
               newImageData.SetScalarComponentFromDouble(i, j, k, 0, c_out[i,j,k])

   # Apply marching cubes algorithm
   marching = vtk.vtkMarchingCubes()
   marching.SetInputData(newImageData)
   marching.SetValue(0, -550)
   marching.Update()
   # Smooth the surface with the windowed sinc filter
   lineStrip = vtk.vtkStripper()
   lineStrip.SetInputConnection(marching.GetOutputPort())
   smoother = vtk.vtkWindowedSincPolyDataFilter()
   smoother.SetInputConnection(lineStrip.GetOutputPort())
   smoother.SetNumberOfIterations(200) #100
   smoother.SetPassBand(0.1) #0.2

   print("\nWriting results of extended STL to file...", flush=True)
   # Write to .STL file
   writer = vtk.vtkSTLWriter()
   writer.SetInputConnection(smoother.GetOutputPort())
   writer.SetFileTypeToBinary()
   writer.SetFileName(os.path.join(output_dir, output_file_ext))
   writer.Write()

   if force_out_cut==True:

      cT_stl = STLReader(FileNames=[os.path.join(output_dir, output_file_ext)])
      cT_stl.UpdatePipeline()

      clip = Clip(Input=cT_stl)
      clip.ClipType = "Plane"
      clip.Scalars = ["CELLS", "STLSolidLabeling"]

      clip.ClipType.Origin = [force_OUT_c[0], force_OUT_c[1], force_OUT_c[2]]
      clip.ClipType.Normal = [force_OUT_n[0], force_OUT_n[1], force_OUT_n[2]]
      clip.Invert = 1

      extractSurface = ExtractSurface(Input=clip)
      triangulate = Triangulate(Input=extractSurface)

      SaveData(os.path.join(output_dir, output_file_ext), proxy=triangulate, FileType="Ascii")
   
   print("Segmentation extension done.\n")
