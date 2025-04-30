import os
import sys
import numpy as np
import pydicom
import SimpleITK as sitk
import tensorflow as tf
import vtk
import nrrd
import time

from scipy import ndimage
from scipy.ndimage import zoom

from tensorflow.keras.layers import (
    Input
)

from cython0.cython_voxel_layer import voxel_layer
from cython2.cython_extend_boundary import extend_boundary
from cython3.cython_reduce_boundary import reduce_boundary

from omegaconf import DictConfig, OmegaConf
import hydra

from conv import get_net_2D, get_net_3D, conv2d_block, conv3d_block, getLargestCC

@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # Start of segmentaion
    print("#" * 30)
    print("Segmentation")
    print("#" * 30)

    start_step_i = time.time()

    if cfg.dicom.use == True:

        # Load the DICOM files
        print("Loading DICOM files...", flush=True)
        files = []
        for file in os.listdir(cfg.path.data):
            file_path = os.path.join(cfg.path.data, file)
            # print("file path: ", ""+str(file_path)+"", flush=True)
            if os.path.isfile(file_path):
                try:
                    ds = pydicom.filereader.dcmread(file_path)
                    if cfg.boundary.enable_series_number:
                        # print("series number: ", ds.SeriesNumber, flush=True)
                        if ds.SeriesNumber == cfg.boundary.series_number:  # and ds.SeriesTime=='085446':
                            files.append(pydicom.dcmread(file_path))
                            # files.append(dicom.read_file(""+str(file_path)+""))
                    else:
                        files.append(pydicom.dcmread(file_path))
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
        # print("Spacing between slices: %f" % slices[0].SpacingBetweenSlices, flush=True)
        print(
            "Pixel spacing (row, col): ({}, {})".format(
                slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]
            ),
            flush=True,
        )

        # Write pixel spacing and spacing between slices
        voxel_size = np.zeros((3))
        voxel_size[0] = float(slices[0].PixelSpacing[0])
        voxel_size[1] = float(slices[0].PixelSpacing[1])
        # voxel_size[2]=float(slices[0].SliceThickness)
        voxel_size[2] = float(cfg.dicom.VS_Z)
        print("voxel_size[2]: ", voxel_size[2], flush=True)

        # Create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.empty(img_shape)

        # Fill 3D array with the images from the files
        for i, s in enumerate(slices):
            img3d[:, :, i] = s.pixel_array  # pixel_array = img2d

        # Reshape 3D array and recalculate voxel size if x,y dimensions are not 512x512
        # X axis
        if slices[0].pixel_array.shape[0] != 512:
            x_res = np.round((512 / slices[0].pixel_array.shape[0]), decimals=3)
            img3d = zoom(img3d, (x_res, 1, 1))

            voxel_size[0] = (voxel_size[0] / slices[0].pixel_array.shape[0]) * 512

        # Y axis
        if slices[0].pixel_array.shape[1] != 512:
            y_res = np.round((512 / slices[0].pixel_array.shape[1]), decimals=3)
            img3d = zoom(img3d, (1, y_res, 1))

            voxel_size[1] = (voxel_size[1] / slices[0].pixel_array.shape[1]) * 512

        # Swap y-axis, to match with CT data
        # img3d_rev = np.empty((img3d.shape[0],img3d.shape[1],img3d.shape[2]))
        # for i in range(img3d.shape[1]):
        #    img3d_rev[:,i,:]=img3d[:,(-i+(img3d.shape[1]-1)),:]

        # img3d=img3d_rev

        # If necessary, convert from CT values to HU units
        RS = slices[0].RescaleSlope
        RI = slices[0].RescaleIntercept
        print("\nRS: ", RS, flush=True)
        print("\nRI: ", RI, flush=True)


    else:

        # read external dicom
        img3d, header = nrrd.read(os.path.join(cfg.path.results, cfg.dicom.ext_file))
        # Write pixel spacing and spacing between slices
        voxel_size = np.zeros((3))
        voxel_size[0] = float(cfg.dicom.VS_X)
        voxel_size[1] = float(cfg.dicom.VS_Y)
        # voxel_size[2]=float(slices[0].SliceThickness)
        voxel_size[2] = float(cfg.dicom.VS_Z)

        RS = cfg.dicom.ext_RS
        RI = cfg.dicom.ext_RI

    nrrd.write(os.path.join(cfg.path.results, cfg.path.voxel_size), voxel_size)

    img3d = img3d * RS + RI

    print("\nSetting up network...", flush=True)
    # Kernel for convolution filter
    k = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    # Initialisation of output file
    CONV_and_GAD_filter = np.empty((img3d.shape[0], img3d.shape[1], img3d.shape[2]))

    for i in range(img3d.shape[0]):
        slice_conv_filter = ndimage.convolve(img3d[i, :, :], k, mode="constant", cval=0)
        slice_GAD = sitk.GetImageFromArray(slice_conv_filter)
        slice_GAD = sitk.Cast(slice_GAD, sitk.sitkFloat32)
        GAD = sitk.GradientAnisotropicDiffusionImageFilter()
        GAD.SetNumberOfIterations(8)
        GAD.SetTimeStep(0.0625)
        GAD.SetConductanceParameter(5.0)
        filtered_slice = GAD.Execute(slice_GAD)
        CONV_and_GAD_filter[i, :, :] = sitk.GetArrayFromImage(filtered_slice)


    # Write out raw CT data as array
    if cfg.boundary.write_ct == True:
        nrrd.write(os.path.join(cfg.path.results, cfg.path.ct), img3d)
    # Write out filtered CT data as array
    if cfg.boundary.write_filtered == True:
        nrrd.write(os.path.join(cfg.path.results, cfg.path.filtered_ct), CONV_and_GAD_filter)

    #######################
    # CNN-A
    #######################

    # Define input
    X = CONV_and_GAD_filter
    # Normalize input
    X_norm_1 = (X - cfg.cnnA.x_mean) / cfg.cnnA.x_std
    # Reshape test data to match with code later
    X_norm_res_1 = np.empty((X_norm_1.shape[2], 512, 512, 1))

    for i in range(X_norm_1.shape[2]):
        X_norm_res_1[i, :, :, 0] = X_norm_1[:, :, i]

    # Initialize network for segmentation
    print("Initializing network for segmentation...", flush=True)
    input_img_1 = Input((512, 512, 1), name="img")
    model_1 = get_net_2D(input_img_1, out_layer=1, n_filters=32, dropout=0, batchnorm=True)
    # model.summary()

    # Load weights and biases of trained network for segmentation
    model_1.load_weights("./seg_A.h5")

    print("Starting segmentation...", flush=True)
    # Start segmentation
    segmentation = model_1.predict(X_norm_res_1, verbose=1)
    # Set pixel quantities to binary values 0 or 1 around threshold 0.5
    segmentation[segmentation > 0.5] = 1.0
    segmentation[segmentation <= 0.5] = 0
    # Remove 4th dimension of segmentation
    segmentation_res = np.zeros((X.shape[2], X.shape[0], X.shape[1]))
    for i in range(X.shape[2]):
        segmentation_res[i] = segmentation[i, :, :, 0]
    # Transpose segmentation to x,y,z coordinates
    segmentation_tra = np.transpose(segmentation_res, (1, 2, 0))

    # Keep largest island
    if cfg.boundary.KLI == True:
        segmentation_tra = getLargestCC(segmentation_tra)

    # Save as .nrrd file
    nrrd.write(os.path.join(cfg.path.results, cfg.path.segmentation), segmentation_tra)

    #######################
    # CNN-B
    #######################

    # Identify coordinates of first segmented voxel in x-direction
    # Based on that, nostril dice is refined and thin inlet layers are predicted
    # +/- a certain number of slices in z-direction around that point

    # Loop through the first 1/3 of the segmentation in x-direction
    # to find the spanwidth in z-direction between frontal sinuses and nostrils

    span_up = 0
    span_down = segmentation_tra.shape[2]

    # Loop through segmentation
    for i in range(int(segmentation_tra.shape[0] / 3)):
        for j in range(segmentation_tra.shape[1]):
            for k in range(segmentation_tra.shape[2]):

                # Update span_up and span_down
                if segmentation_tra[i, j, k] == 1 and k > span_up:

                    span_up = k

                if segmentation_tra[i, j, k] == 1 and k < span_down:

                    span_down = k

    print("span_up: ", span_up)
    print("span_down: ", span_down)

    # Save as .nrrd file
    span = np.zeros((2))
    span[0] = span_up
    span[1] = span_down
    nrrd.write(os.path.join(cfg.path.results, cfg.path.span), span)

    # Loop through segmentation from 0 to the half spanwidth in z-direction
    # to find minimum x coordinate of nostrils
    flag = True

    for i in range(segmentation_tra.shape[0]):
        for j in range(segmentation_tra.shape[1]):
            for k in range(int(span_down + (span_up - span_down) / 2)):

                # When finding the first segmented voxel, break all loops and store the z position
                if segmentation_tra[i, j, k] == 1:

                    x_min_coord = [i, j, k]

                    flag = False
                    break

            if flag == False:

                break

        if flag == False:

            break

    print("x_min_coord: ", x_min_coord, flush=True)

    # Dice from filtered CT data: dice_X
    # Check if dice exceeds z-bounds of CT data
    if x_min_coord[2] - cfg.cnnB.z_back < 0:

        dice_X = X[
            int(x_min_coord[0] - cfg.cnnB.x_back) : int(x_min_coord[0] + cfg.cnnB.x_front),
            int(x_min_coord[1] - cfg.cnnB.y_back) : int(x_min_coord[1] + cfg.cnnB.y_front),
            0:cfg.cnnB.z_edge,
        ]

    else:

        dice_X = X[
            int(x_min_coord[0] - cfg.cnnB.x_back) : int(x_min_coord[0] + cfg.cnnB.x_front),
            int(x_min_coord[1] - cfg.cnnB.y_back) : int(x_min_coord[1] + cfg.cnnB.y_front),
            int(x_min_coord[2] - cfg.cnnB.z_back) : int(x_min_coord[2] + cfg.cnnB.z_front),
        ]


    # Normalize input
    X_norm_2 = (dice_X - cfg.cnnB.x_mean) / cfg.cnnB.x_std

    # Reshape test data to match with code later
    X_norm_res_2 = np.zeros((1, X_norm_2.shape[0], X_norm_2.shape[1], X_norm_2.shape[2], 1))

    X_norm_res_2[0, :, :, :, 0] = X_norm_2[:, :, :]

    # Initialize network for dice
    print("Initializing network for prediction of dice around nostrils...", flush=True)
    # Input
    input_img_2 = Input((cfg.cnnB.x_edge, cfg.cnnB.y_edge, cfg.cnnB.z_edge, 1), name="img")

    # Initialize network
    model_2 = get_net_3D(input_img_2, n_filters=32, dropout=0, batchnorm=True)

    # Load weights and biases of trained network for segmentation
    model_2.load_weights("./seg_B.h5")

    print("Starting prediction of dice around nostrils...", flush=True)
    # Start prediction
    dice = model_2.predict(X_norm_res_2, verbose=1)

    # Set pixel quantities to binary values 0 or 1 around threshold 0.5
    dice[dice > 0.5] = 1.0
    dice[dice <= 0.5] = 0

    # Remove 1st and 4th dimension of dice
    dice_res = np.zeros((dice.shape[1], dice.shape[2], dice.shape[3]))

    dice_res = dice[0, :, :, :, 0]

    # Save as .nrrd file
    nrrd.write(os.path.join(cfg.path.results, cfg.path.dice), dice_res)

    # Fill segmentation with predicted dice
    if x_min_coord[2] - cfg.cnnB.z_back < 0:

        segmentation_tra[
            int(x_min_coord[0] - cfg.cnnB.x_back) : int(x_min_coord[0] + cfg.cnnB.x_front),
            int(x_min_coord[1] - cfg.cnnB.y_back) : int(x_min_coord[1] + cfg.cnnB.y_front),
            0:cfg.cnnB.z_edge,
        ] = dice_res

    else:

        segmentation_tra[
            int(x_min_coord[0] - cfg.cnnB.x_back) : int(x_min_coord[0] + cfg.cnnB.x_front),
            int(x_min_coord[1] - cfg.cnnB.y_back) : int(x_min_coord[1] + cfg.cnnB.y_front),
            int(x_min_coord[2] - cfg.cnnB.z_back) : int(x_min_coord[2] + cfg.cnnB.z_front),
        ] = dice_res

    # Keep largest island
    segmentation_tra = getLargestCC(segmentation_tra)

    # Save as .nrrd file
    nrrd.write(os.path.join(cfg.path.results, cfg.path.seg_dice), segmentation_tra)

    end_step_i = time.time()
    print("step (i)")
    print("{:5.3f}s".format(end_step_i - start_step_i))

    #######################
    # CNN-C
    #######################

    start_step_iii_1 = time.time()

    # Identify last segmented voxel in x-direction,
    # to calculate the x bound of the nasal cavity
    flag = True

    # Loop through segmentation
    for i in range(segmentation_tra.shape[0] - 1, 0, -1):
        for j in range(segmentation_tra.shape[1]):
            for k in range(segmentation_tra.shape[2]):

                # When finding the first segmented voxel, break all loops and store the z position
                if segmentation_tra[i, j, k] == 1:

                    flag = False

                    x_max_coord = [i, j, k]

                    break

            if flag == False:

                break

        if flag == False:

            break

    # x bound of nasal cavity
    L_x = (x_max_coord[0] - x_min_coord[0]) * voxel_size[0]

    # Nr of slices in each z-direction to predict thin inlet layers
    nr_sl_z = int((cfg.outlet.z_dist_norm / voxel_size[2]) * L_x)

    # Initialize network for left inlet
    print("Initializing network for inlets...", flush=True)
    input_img_3 = Input((512, 512, 2), name="img")
    model_3 = get_net_2D(input_img_3, out_layer=2, n_filters=32, dropout=0, batchnorm=True)
    # model.summary()

    # Load weights and biases of trained network for left inlet
    model_3.load_weights("./seg_C.h5")

    # Reshape and normalize data
    X_norm_res_3 = np.empty((2 * nr_sl_z, 512, 512, 2))

    print("nr_sl_z: ", nr_sl_z, flush=True)

    # Check if x_min_coord is to close to the min_z of the CT data
    if (x_min_coord[2] - nr_sl_z) < 0:

        for i in range(2 * nr_sl_z):

            X_norm_res_3[i, :, :, 0] = (X[:, :, i] - cfg.cnnC.x_mean) / cfg.cnnC.x_std
            X_norm_res_3[i, :, :, 1] = segmentation_tra[:, :, i]

    else:

        for i in range(x_min_coord[2] - nr_sl_z, x_min_coord[2] + nr_sl_z):

            X_norm_res_3[i + nr_sl_z - x_min_coord[2], :, :, 0] = (
                X[:, :, i] - cfg.cnnC.x_mean
            ) / cfg.cnnC.x_std
            X_norm_res_3[i + nr_sl_z - x_min_coord[2], :, :, 1] = segmentation_tra[:, :, i]

    print("Starting prediction of inlets...", flush=True)
    # Start prediction of left inlet
    inlets = model_3.predict(X_norm_res_3, verbose=1)
    # Set pixel quantities to binary values 0 or 1 around threshold 0.5
    inlets[inlets > 0.5] = 1.0
    inlets[inlets <= 0.5] = 0

    # Remove 4th dimension of inlets
    inlet_left = np.zeros((2 * nr_sl_z, X.shape[0], X.shape[1]))
    inlet_right = np.zeros((2 * nr_sl_z, X.shape[0], X.shape[1]))

    for i in range(2 * nr_sl_z):
        inlet_left[i] = inlets[i, :, :, 0]
        inlet_right[i] = inlets[i, :, :, 1]

    # Transpose to x,y,z coordinates
    inlet_left_tra = np.transpose(inlet_left, (1, 2, 0))
    inlet_right_tra = np.transpose(inlet_right, (1, 2, 0))

    # Fill complete array
    if (x_min_coord[2] - nr_sl_z) < 0:

        inlet_left_full = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_left_full[:, :, 0 : 2 * nr_sl_z] = inlet_left_tra

        inlet_right_full = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_right_full[:, :, 0 : 2 * nr_sl_z] = inlet_right_tra

    else:

        inlet_left_full = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_left_full[:, :, x_min_coord[2] - nr_sl_z : x_min_coord[2] + nr_sl_z] = (
            inlet_left_tra
        )

        inlet_right_full = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_right_full[:, :, x_min_coord[2] - nr_sl_z : x_min_coord[2] + nr_sl_z] = (
            inlet_right_tra
        )

    # Keep largest island
    inlet_left_full = getLargestCC(inlet_left_full)
    inlet_right_full = getLargestCC(inlet_right_full)

    # Save as .nrrd file
    nrrd.write(os.path.join(cfg.path.results, cfg.path.inlet_left), inlet_left_full)
    nrrd.write(os.path.join(cfg.path.results, cfg.path.inlet_right), inlet_right_full)

    end_step_iii_1 = time.time()
    print("step (iii) 1")
    print("{:5.3f}s".format(end_step_iii_1 - start_step_iii_1))

    start_step_ii = time.time()
    # print("\nFilling holes and create new image data...", flush=True)
    print("\nCreating new image data......", flush=True)
    # gt_binary = ndimage.binary_fill_holes(segmentation_tra).astype(int)

    # Use external segmentation
    if cfg.external.use == True:

        segmentation_tra, header = nrrd.read(os.path.join(cfg.path.results, cfg.external.name))


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
    # Generate 2 voxel layer array
    c_out = voxel_layer(b_input, a_input, c_input)

    # Create a new vtk ImageData that contains the previously generated array.
    # Here, it is important to use pixel dimensions and pixel spacing
    # from the original CT image (!).
    newImageData = vtk.vtkImageData()
    newImageData.SetDimensions(c.shape[0], c.shape[1], c.shape[2])
    newImageData.SetSpacing(
        voxel_size
        # (float(slices[0].PixelSpacing[0])*0.001,
        # float(slices[0].PixelSpacing[1])*0.001,
        # float(slices[0].SliceThickness)*0.001)
    )
    newImageData.origin = (0, 0, 0)
    newImageData.AllocateScalars(vtk.VTK_INT, 1)

    for k in range(c.shape[2]):
        for j in range(c.shape[1]):
            for i in range(c.shape[0]):
                newImageData.SetScalarComponentFromDouble(i, j, k, 0, c_out[i, j, k])

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
    smoother.SetNumberOfIterations(200)  # 100
    smoother.SetPassBand(0.1)  # 0.2

    print("\nWriting results to file...", flush=True)
    # Write to .STL file
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(os.path.join(cfg.path.results, cfg.path.output))
    writer.Write()

    if cfg.outlet.force_cut == True:

        cT_stl = STLReader(FileNames=[os.path.join(cfg.path.results, cfg.path.output)])
        cT_stl.UpdatePipeline()

        clip = Clip(Input=cT_stl)
        clip.ClipType = "Plane"
        clip.Scalars = ["CELLS", "STLSolidLabeling"]

        clip.ClipType.Origin = [force_OUT_c[0], force_OUT_c[1], force_OUT_c[2]]
        clip.ClipType.Normal = [force_OUT_n[0], force_OUT_n[1], force_OUT_n[2]]
        clip.Invert = 1

        extractSurface = ExtractSurface(Input=clip)
        triangulate = Triangulate(Input=extractSurface)

        SaveData(os.path.join(cfg.path.results, cfg.path.output), proxy=triangulate, FileType="Ascii")

    print("Segmentation done.\n")

    end_step_ii = time.time()
    print("step (ii)")
    print("{:5.3f}s".format(end_step_ii - start_step_ii))

    # Boundary extension
    if cfg.boundary.seg_ext == True or cfg.boundary.seg_red == True:

        SEG = segmentation_tra.astype(float)
        SEG_NEW = segmentation_tra.astype(float)

        # loop over number of extensions
        for i in range(cfg.boundary.seg_ext_n):

            SEG = SEG.astype(float)
            SEG_NEW = SEG_NEW.astype(float)

            if cfg.boundary.seg_ext == True:
                SEG = extend_boundary(SEG, SEG_NEW)
            elif cfg.boundary.seg_red == True:
                SEG = reduce_boundary(SEG, SEG_NEW)

            SEG_NEW = SEG

        # Keep largest island
        if cfg.boundary.KLI == True:
            SEG = getLargestCC(SEG)

        # write extended segmentation
        nrrd.write(os.path.join(cfg.path.results, cfg.path.seg_ext), SEG)

        # Initialize 2 voxel layer array
        c = (SEG * -100000) + 50000

        # Convert datatype to match with Cython code
        c_input = c.astype(float)
        a_input = CONV_and_GAD_filter.astype(float)
        b_input = SEG.astype(float)
        # Generate 2 voxel layer array
        c_out = voxel_layer(b_input, a_input, c_input)

        # Create a new vtk ImageData that contains the previously generated array.
        # Here, it is important to use pixel dimensions and pixel spacing
        # from the original CT image (!).
        newImageData = vtk.vtkImageData()
        newImageData.SetDimensions(c.shape[0], c.shape[1], c.shape[2])
        newImageData.SetSpacing(
            voxel_size
            # (float(slices[0].PixelSpacing[0])*0.001,
            # float(slices[0].PixelSpacing[1])*0.001,
            # float(slices[0].SliceThickness)*0.001)
        )
        newImageData.origin = (0, 0, 0)
        newImageData.AllocateScalars(vtk.VTK_INT, 1)

        for k in range(c.shape[2]):
            for j in range(c.shape[1]):
                for i in range(c.shape[0]):
                    newImageData.SetScalarComponentFromDouble(i, j, k, 0, c_out[i, j, k])

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
        smoother.SetNumberOfIterations(200)  # 100
        smoother.SetPassBand(0.1)  # 0.2

        print("\nWriting results of extended STL to file...", flush=True)
        # Write to .STL file
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(smoother.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileName(os.path.join(cfg.path.results, cfg.path.output_ext))
        writer.Write()

        if cfg.outlet.force_cut == True:

            cT_stl = STLReader(FileNames=[os.path.join(cfg.path.results, cfg.path.output_ext)])
            cT_stl.UpdatePipeline()

            clip = Clip(Input=cT_stl)
            clip.ClipType = "Plane"
            clip.Scalars = ["CELLS", "STLSolidLabeling"]

            clip.ClipType.Origin = cfg.outlet.out_c
            clip.ClipType.Normal = cfg.outlet.out_n
            clip.Invert = 1

            extractSurface = ExtractSurface(Input=clip)
            triangulate = Triangulate(Input=extractSurface)

            SaveData(
                os.path.join(cfg.path.results, cfg.path.output_ext),
                proxy=triangulate,
                FileType="Ascii",
            )

        print("Segmentation extension done.\n")


if __name__ == "__main__":
    main()

