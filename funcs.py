import os
import sys
import numpy as np
import pydicom
import SimpleITK as sitk
import vtk
import nrrd
import trimesh

from scipy import ndimage
from scipy.ndimage import zoom

from tensorflow.keras.layers import Input

sys.path.append("cython")
from cython0.cython_voxel_layer import voxel_layer
from cython2.cython_extend_boundary import extend_boundary
from cython3.cython_reduce_boundary import reduce_boundary

from omegaconf import DictConfig, OmegaConf

from cnn.conv import get_net_2D, get_net_3D, getLargestCC

from tensorflow.keras.models import Model


def segmentation_step(config: DictConfig) -> np.ndarray | np.ndarray:
    # Start of segmentation
    print("#" * 30)
    print("Segmentation")
    print("#" * 30)

    """1. Read DICOM Data and sort the CT scans/slices"""

    if config.dicom.use == True:
        # Load the DICOM files
        print("Loading DICOM files...", flush=True)
        files = []
        for file in os.listdir(config.path.data):
            file_path = os.path.join(config.path.data, file)
            # print("file path: ", ""+str(file_path)+"", flush=True)
            if os.path.isfile(file_path):
                try:
                    ds = pydicom.filereader.dcmread(file_path)
                    if config.dicom.enable_series_number:
                        # print("series number: ", ds.SeriesNumber, flush=True)
                        if (
                            ds.SeriesNumber == config.dicom.series_number
                        ):  # and ds.SeriesTime=='085446':
                            files.append(pydicom.dcmread(file_path))
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
        voxel_size[2] = float(config.dicom.VS_Z)
        print("voxel_size[2]: ", voxel_size[2], flush=True)

        # Create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        image = np.empty(img_shape)

        # Fill 3D array with the images from the files
        for i, s in enumerate(slices):
            image[:, :, i] = s.pixel_array  # pixel_array = img2d

        # Reshape 3D array and recalculate voxel size if x,y dimensions are not 512x512
        # X axis
        if slices[0].pixel_array.shape[0] != 512:
            x_res = np.round((512 / slices[0].pixel_array.shape[0]), decimals=3)
            image = zoom(image, (x_res, 1, 1))
            voxel_size[0] = (voxel_size[0] / slices[0].pixel_array.shape[0]) * 512
        # Y axis
        if slices[0].pixel_array.shape[1] != 512:
            y_res = np.round((512 / slices[0].pixel_array.shape[1]), decimals=3)
            image = zoom(image, (1, y_res, 1))
            voxel_size[1] = (voxel_size[1] / slices[0].pixel_array.shape[1]) * 512

        # Swap y-axis, to match with CT data
        # image_rev = np.empty((image.shape[0],image.shape[1],image.shape[2]))
        # for i in range(image.shape[1]):
        #    image_rev[:,i,:]=image[:,(-i+(image.shape[1]-1)),:]

        # image=image_rev

        # If necessary, convert from CT values to HU units
        RS = slices[0].RescaleSlope
        RI = slices[0].RescaleIntercept
        print("\nRS: ", RS, flush=True)
        print("\nRI: ", RI, flush=True)

    else:

        # read external dicom
        image, header = nrrd.read(
            os.path.join(config.path.results, config.dicom.ext_file)
        )
        # Write pixel spacing and spacing between slices
        voxel_size = np.zeros((3))
        voxel_size[0] = float(config.dicom.VS_X)
        voxel_size[1] = float(config.dicom.VS_Y)
        voxel_size[2] = float(config.dicom.VS_Z)

        RS = config.dicom.ext_RS
        RI = config.dicom.ext_RI

    os.makedirs(config.path.results, exist_ok=True)
    nrrd.write(os.path.join(config.path.results, config.path.voxel_size), voxel_size)

    image = image * RS + RI

    return image, voxel_size

def preprocessing_step(
    config: DictConfig, image: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """2. Initialize network set-up"""

    print("\nSetting up network...", flush=True)

    # Initialisation of output file
    X = np.empty((image.shape[0], image.shape[1], image.shape[2]))

    for i in range(image.shape[0]):
        slice_conv_filter = ndimage.convolve(image[i, :, :], k, mode="constant", cval=0)
        slice_GAD = sitk.GetImageFromArray(slice_conv_filter)
        slice_GAD = sitk.Cast(slice_GAD, sitk.sitkFloat32)
        GAD = sitk.GradientAnisotropicDiffusionImageFilter()
        GAD.SetNumberOfIterations(8)
        GAD.SetTimeStep(0.0625)
        GAD.SetConductanceParameter(5.0)
        filtered_slice = GAD.Execute(slice_GAD)
        X[i, :, :] = sitk.GetArrayFromImage(filtered_slice)

    # Write out raw CT data as array
    if config.boundary.write_ct == True:
        nrrd.write(os.path.join(config.path.results, config.path.ct), image)
    # Write out filtered CT data as array
    if config.boundary.write_filtered == True:
        nrrd.write(os.path.join(config.path.results, config.path.filtered_ct), X)

    return X

def CNN_A(
    config: DictConfig, 
    X: np.ndarray
) -> np.ndarray | np.ndarray | Model | np.ndarray:
    """3. CNN-A"""
    # Normalize input
    A_norm = (X - config.cnnA.x_mean) / config.cnnA.x_std
    # Reshape test data to match with code later
    A_norm_res = np.empty((A_norm.shape[2], 512, 512, 1))

    for i in range(A_norm.shape[2]):
        A_norm_res[i, :, :, 0] = A_norm[:, :, i]

    # Initialize network for segmentation
    print("Initializing network for segmentation...", flush=True)
    input_img_1 = Input((512, 512, 1), name="img")
    model_A = get_net_2D(
        input_img_1, out_layer=1, n_filters=32, dropout=0, batchnorm=True
    )
    # model.summary()

    # Load weights and biases of trained network for segmentation
    model_A.load_weights(config.cnnA.path)

    print("Starting segmentation...", flush=True)
    # Start segmentation
    segmentation = model_A.predict(A_norm_res, verbose=1)
    # Set pixel quantities to binary values 0 or 1 around threshold 0.5
    segmentation[segmentation > 0.5] = 1.0
    segmentation[segmentation <= 0.5] = 0
    # Remove 4th dimension of segmentation
    segmentation_res = np.zeros((X.shape[2], X.shape[0], X.shape[1]))
    for i in range(X.shape[2]):
        segmentation_res[i] = segmentation[i, :, :, 0]
    # Transpose segmentation to x,y,z coordinates
    segmentation_tra = np.transpose(segmentation_res, (1, 2, 0))

    segmentation = segmentation_tra
    # Keep largest island
    if config.boundary.KLI == True:
        segmentation = getLargestCC(segmentation)

    # Save as .nrrd file
    nrrd.write(
        os.path.join(config.path.results, config.path.segmentation), segmentation
    )

    return A_norm, A_norm_res, model_A, segmentation

def CNN_B(
    config: DictConfig, 
    X: np.ndarray, 
    segmentation: np.ndarray
    ) -> np.ndarray | np.ndarray | Model | list:
    """Identify coordinates of first segmented voxel in x-direction
    Based on that, nostril dice is refined and thin inlet layers are predicted
    +/- a certain number of slices in z-direction around that point
    Loop through the first 1/3 of the segmentation in x-direction
    to find the spanwidth in z-direction between frontal sinuses and nostrils"""

    span_up = 0
    span_down = segmentation.shape[2]

    # Loop through segmentation
    for i in range(int(segmentation.shape[0] / 3)):
        for j in range(segmentation.shape[1]):
            for k in range(segmentation.shape[2]):
                # Update span_up and span_down
                if segmentation[i, j, k] == 1 and k > span_up:
                    span_up = k
                if segmentation[i, j, k] == 1 and k < span_down:
                    span_down = k
    print("span_up: ", span_up)
    print("span_down: ", span_down)

    # Save as .nrrd file
    span = np.zeros((2))
    span[0] = span_up
    span[1] = span_down
    nrrd.write(os.path.join(config.path.results, config.path.span), span)

    # Loop through segmentation from 0 to the half spanwidth in z-direction
    # to find minimum x coordinate of nostrils
    flag = True

    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            for k in range(int(span_down + (span_up - span_down) / 2)):
                # When finding the first segmented voxel, break all loops and store the z position
                if segmentation[i, j, k] == 1:
                    min_coord = [i, j, k]
                    flag = False
                    break
            if flag == False:
                break
        if flag == False:
            break

    print("min_coord: ", min_coord, flush=True)

    # Dice from filtered CT data: dice_X
    # Check if dice exceeds z-bounds of CT data
    if min_coord[2] - config.cnnB.z_back < 0:
        dice_X = X[
            int(min_coord[0] - config.cnnB.x_back) : int(
                min_coord[0] + config.cnnB.x_front
            ),
            int(min_coord[1] - config.cnnB.y_back) : int(
                min_coord[1] + config.cnnB.y_front
            ),
            0 : config.cnnB.z_edge,
        ]
    else:
        dice_X = X[
            int(min_coord[0] - config.cnnB.x_back) : int(
                min_coord[0] + config.cnnB.x_front
            ),
            int(min_coord[1] - config.cnnB.y_back) : int(
                min_coord[1] + config.cnnB.y_front
            ),
            int(min_coord[2] - config.cnnB.z_back) : int(
                min_coord[2] + config.cnnB.z_front
            ),
        ]

    # Normalize input
    B_norm = (dice_X - config.cnnB.x_mean) / config.cnnB.x_std

    # Reshape test data to match with code later
    B_norm_res = np.zeros((1, B_norm.shape[0], B_norm.shape[1], B_norm.shape[2], 1))
    B_norm_res[0, :, :, :, 0] = B_norm[:, :, :]

    # Initialize network for dice
    print("Initializing network for prediction of dice around nostrils...", flush=True)
    # Input
    input_img_2 = Input(
        (config.cnnB.x_edge, config.cnnB.y_edge, config.cnnB.z_edge, 1), name="img"
    )
    # Initialize network
    model_B = get_net_3D(input_img_2, n_filters=32, dropout=0, batchnorm=True)
    # Load weights and biases of trained network for segmentation
    model_B.load_weights(config.cnnB.path)
    print("Starting prediction of dice around nostrils...", flush=True)
    # Start prediction
    dice = model_B.predict(B_norm_res, verbose=1)
    # Set pixel quantities to binary values 0 or 1 around threshold 0.5
    dice[dice > 0.5] = 1.0
    dice[dice <= 0.5] = 0
    # Remove 1st and 4th dimension of dice
    dice_res = np.zeros((dice.shape[1], dice.shape[2], dice.shape[3]))
    dice_res = dice[0, :, :, :, 0]
    # Save as .nrrd file
    nrrd.write(os.path.join(config.path.results, config.path.dice), dice_res)
    # Fill segmentation with predicted dice
    if min_coord[2] - config.cnnB.z_back < 0:
        dice_res = segmentation[
            int(min_coord[0] - config.cnnB.x_back) : int(
                min_coord[0] + config.cnnB.x_front
            ),
            int(min_coord[1] - config.cnnB.y_back) : int(
                min_coord[1] + config.cnnB.y_front
            ),
            0 : config.cnnB.z_edge,
        ]
    else:
        dice_res = segmentation[
            int(min_coord[0] - config.cnnB.x_back) : int(
                min_coord[0] + config.cnnB.x_front
            ),
            int(min_coord[1] - config.cnnB.y_back) : int(
                min_coord[1] + config.cnnB.y_front
            ),
            int(min_coord[2] - config.cnnB.z_back) : int(
                min_coord[2] + config.cnnB.z_front
            ),
        ]

    # Save as .nrrd file
    nrrd.write(os.path.join(config.path.results, config.path.seg_dice), segmentation)

    return B_norm, B_norm_res, model_B, min_coord

def CNN_C(
    config: DictConfig,
    X: np.ndarray,
    segmentation: np.ndarray,
    min_coord: list,
    voxel_size: np.ndarray,
) -> np.ndarray | Model | np.ndarray | np.ndarray:

    # Identify last segmented voxel in x-direction,
    # to calculate the x bound of the nasal cavity
    flag = True
    # Loop through segmentation
    for i in range(segmentation.shape[0] - 1, 0, -1):
        for j in range(segmentation.shape[1]):
            for k in range(segmentation.shape[2]):
                # When finding the first segmented voxel, break all loops and store the z position
                if segmentation[i, j, k] == 1:
                    flag = False
                    x_max_coord = [i, j, k]
                    break
            if flag == False:
                break
        if flag == False:
            break
    # x bound of nasal cavity
    L_x = (x_max_coord[0] - min_coord[0]) * voxel_size[0]
    # Nr of slices in each z-direction to predict thin inlet layers
    nr_sl_z = int((config.outlet.z_dist_norm / voxel_size[2]) * L_x)
    # Initialize network for left inlet
    print("Initializing network for inlets...", flush=True)
    input_img_3 = Input((512, 512, 2), name="img")
    model_C = get_net_2D(
        input_img_3, out_layer=2, n_filters=32, dropout=0, batchnorm=True
    )
    # model.summary()
    # Load weights and biases of trained network for left inlet
    model_C.load_weights(config.cnnC.path)
    # Reshape and normalize data
    C_norm_res = np.empty((2 * nr_sl_z, 512, 512, 2))

    print("nr_sl_z: ", nr_sl_z, flush=True)

    # Check if min_coord is to close to the min_z of the CT data
    if (min_coord[2] - nr_sl_z) < 0:
        for i in range(2 * nr_sl_z):
            C_norm_res[i, :, :, 0] = (
                X[:, :, i] - config.cnnC.x_mean
            ) / config.cnnC.x_std
            C_norm_res[i, :, :, 1] = segmentation[:, :, i]
    else:
        for i in range(min_coord[2] - nr_sl_z, min_coord[2] + nr_sl_z):
            C_norm_res[i + nr_sl_z - min_coord[2], :, :, 0] = (
                X[:, :, i] - config.cnnC.x_mean
            ) / config.cnnC.x_std

            C_norm_res[i + nr_sl_z - min_coord[2], :, :, 1] = segmentation[:, :, i]

    print("Starting prediction of inlets...", flush=True)
    # Start prediction of left inlet
    inlets = model_C.predict(C_norm_res, verbose=1)
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
    if (min_coord[2] - nr_sl_z) < 0:
        inlet_left = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_left[:, :, 0 : 2 * nr_sl_z] = inlet_left_tra
        inletinlet_right_right_full = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_right[:, :, 0 : 2 * nr_sl_z] = inlet_right_tra
    else:
        inlet_left = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_left[:, :, min_coord[2] - nr_sl_z : min_coord[2] + nr_sl_z] = (
            inlet_left_tra
        )

        inlet_right = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        inlet_right[:, :, min_coord[2] - nr_sl_z : min_coord[2] + nr_sl_z] = (
            inlet_right_tra
        )

    # Keep largest island
    inlet_left = getLargestCC(inlet_left)
    inlet_right = getLargestCC(inlet_right)

    # Save as .nrrd file
    nrrd.write(os.path.join(config.path.results, config.path.inlet_left), inlet_left)
    nrrd.write(os.path.join(config.path.results, config.path.inlet_right), inlet_right)

    return C_norm_res, model_C, inlet_left, inlet_right

def final_step(
    config: DictConfig,
    X: np.ndarray,
    segmentation: np.ndarray,
    voxel_size: np.ndarray,
) -> None:

    # print("\nFilling holes and create new image data...", flush=True)
    print("\nCreating new image data......", flush=True)
    # gt_binary = ndimage.binary_fill_holes(segmentation).astype(int)

    # Use external segmentation
    if config.external.use == True:
        segmentation, header = nrrd.read(
            os.path.join(config.path.results, config.external.name)
        )

    # Initialize 2 voxel layer array
    c = (segmentation * -100000) + 50000
    # Convert datatype to match with Cython code
    c_input = c.astype(float)
    a_input = X.astype(float)
    b_input = segmentation.astype(float)
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

    ## TODO: give a flag to write only internal STL or both:

    # newImageData = X.astype(float)

    # TODO: set this in a config file 
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
    writer.SetFileName(os.path.join(config.path.results, config.path.output))
    writer.Write()

    if config.outlet.force_cut == True:

        mesh = trimesh.load_mesh(
            os.path.join(config.path.results, config.path.output), force="mesh"
        )

        # Define clipping plane
        origin = np.array(config.outlet.out_c)
        normal = np.array(config.outlet.out_n)
        normal /= -np.linalg.norm(normal)
        clipped = mesh.slice_plane(plane_origin=origin, plane_normal=normal)
        clipped.export(
            os.path.join(config.path.results, config.path.output), file_type="stl"
        )

    print("Segmentation extension done.\n")

    # Boundary extension
    if config.boundary.seg_ext == True or config.boundary.seg_red == True:

        SEG = segmentation.astype(float)
        SEG_NEW = segmentation.astype(float)

        # loop over number of extensions
        for i in range(config.boundary.seg_ext_n):

            SEG = SEG.astype(float)
            SEG_NEW = SEG_NEW.astype(float)

            if config.boundary.seg_ext == True:
                SEG = extend_boundary(SEG, SEG_NEW)
            elif config.boundary.seg_red == True:
                SEG = reduce_boundary(SEG, SEG_NEW)

            SEG_NEW = SEG

        # Keep largest island
        if config.boundary.KLI == True:
            SEG = getLargestCC(SEG)

        # write extended segmentation
        nrrd.write(os.path.join(config.path.results, config.path.seg_ext), SEG)

        # Initialize 2 voxel layer array
        c = (SEG * -100000) + 50000

        # Convert datatype to match with Cython code
        c_input = c.astype(float)
        a_input = X.astype(float)
        b_input = SEG.astype(float)
        # Generate 2 voxel layer array
        c_out = voxel_layer(b_input, a_input, c_input)

        """Create a new vtk ImageData that contains the previously generated array.
        Here, it is important to use pixel dimensions and pixel spacing
        from the original CT image (!). """
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
                    newImageData.SetScalarComponentFromDouble(
                        i, j, k, 0, c_out[i, j, k]
                    )

        # Apply marching cubes algorithm
        # TODO: set in config and reuse
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
        writer.SetFileName(os.path.join(config.path.results, config.path.output_ext))
        writer.Write()
