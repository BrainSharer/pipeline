# .mat are actually dictionnary. This function support .mat from
# antsRegistration that encode a 4x4 transformation matrix.
import ants

transformation = 'Affine'
moving = 'ALLEN771602'
fixed = 'Allen'
um = "10"


mi = ants.image_read(f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/{moving}/{moving}_{um}um_sagittal.tif')
fi = ants.image_read(f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/Allen/{fixed}_{um}um_sagittal.tif')
tx = ants.registration(fixed=fi, moving=mi, type_of_transform = (transformation) )
#arr = tx['warpedmovout']
mywarpedimage = ants.apply_transforms( fixed=fi, moving=mi, transformlist=tx['fwdtransforms'], defaultvalue=0 )
ants.image_write(mywarpedimage, f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/{moving}/{moving}_{fixed}_{um}um_sagittal.tif')

""""
original_filepath = tx['fwdtransforms'][0]
new_filepath = f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/{moving}/{moving}_{um}um_sagittal_to_Allen.mat'
shutil.move(original_filepath, f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/{moving}/{moving}_{um}um_sagittal_to_Allen.mat')


new_filepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ls_to_template_rigid_0GenericAffine.mat'
transfo_dict = loadmat(new_filepath)
lps2ras = np.diag([-1, -1, 1])

for k,v in transfo_dict.items():
    print(k, v)


rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
trans = transfo_dict['AffineTransform_float_3_3'][9:12]
offset = transfo_dict['fixed']
r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]


matrix = np.eye(4)
matrix[0:3, 3] = r_trans
matrix[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)

print(matrix)

translation = (matrix[..., 3][0:3])
#translation = 0
matrix = matrix[:3, :3]
volume_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_25um_sagittal.tif'
volume = read_image(volume_path)
transformed_volume = affine_transform(volume, matrix, offset=translation, order=1)
outpath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_25um_sagittal_transformed.tif'
write_image(outpath, transformed_volume)
print(f'Wrote transformed volume to {outpath}')
"""


