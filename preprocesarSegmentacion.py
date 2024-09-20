import nibabel as nib
import os
import numpy as np

def preprocesar():
    ruta = os.getcwd()
        #Load the NIfTI file
    img = nib.load(f"{ruta}/tomografias_segmentadas/4_Pelvis_Osea_20230418192551_4/4_Pelvis_Osea_20230418192551_4_seg.nii.gz")

    # Get the image data and affine matrix
    data = img.get_fdata()
    affine = img.affine

    #Flip the axial slices (z-axis)
    flipped_data = np.flip(data, axis=1)

    # Get the affine matrix
    affine = img.affine
    # Example: Flip the y-axis
    affine[1, 1] = -affine[1, 1]
    affine[1, 3] = -affine[1, 3]
    # Save the flipped image
    flipped_img = nib.Nifti1Image(flipped_data, affine)
    nib.save(flipped_img,  f"{ruta}/tomografias_segmentadas/4_Pelvis_Osea_20230418192551_4/4_Pelvis_Osea_20230418192551_4_seg_flipendo.nii.gz")

    print("Axial slices flipped and saved to 'flipped_image.nii.gz'")