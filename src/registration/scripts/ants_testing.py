# .mat are actually dictionnary. This function support .mat from
# antsRegistration that encode a 4x4 transformation matrix.
import argparse
import shutil
import ants
import os


def create_registration(um):
    transformation = 'Affine'
    moving = 'ALLEN771602'
    fixed = 'Allen'
    um = str(um)

    PATH = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    moving_path = os.path.join(PATH, moving)
    fixed_path = os.path.join(PATH, fixed)
    fixed_filepath = os.path.join(fixed_path, 'Allen_8x7.2x7.2um_sagittal.tif')
    moving_filepath = os.path.join(moving_path, 'ALLEN771602_8x7.2x7.2um_sagittal.tif')

    if not os.path.isfile(moving_filepath):
        print(f"Normalized image not found at {moving_filepath}.")
        exit(1)
    else:
        print(f"Normalized image found at {moving_filepath}.")


    if not os.path.isfile(fixed_filepath):
        print(f"Fixed image not found at {fixed_path}.")
        exit(1)
    else:
        print(f"Fixed image found at {fixed_filepath}.")
                                    

    moving_image = ants.image_read(moving_filepath)
    fixed_image = ants.image_read(fixed_filepath)
    tx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform = (transformation) )
    print(tx)
    mywarpedimage = ants.apply_transforms( fixed=fixed_image, moving=moving_image, transformlist=tx['fwdtransforms'], defaultvalue=0 )
    outpath = os.path.join(moving_path, f'{moving}_{fixed}_{um}um_sagittal.tif')
    ants.image_write(mywarpedimage, outpath)

    original_filepath = tx['fwdtransforms'][0]
    transform_filepath = os.path.join(moving_path, f'{moving}_{fixed}_{um}um_sagittal_to_Allen.mat')
    shutil.move(original_filepath, transform_filepath)


def create_matrix():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument("--um", help="Enter um", required=True, default=10, type=int)
    
    args = parser.parse_args()
    um = args.um
    create_registration(um)
