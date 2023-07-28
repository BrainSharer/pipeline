#!/usr/bin/env python
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import SimpleITK as sitk
import airlab as al
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from toolbox.airlab import dump_al_affine_transform
from toolbox.airlab import load_al_affine_transform

ROOT = Path(__file__).resolve().parent.parent

def load_and_prep_images(
    mov_img_dir, mov_img_thumb_dir,
    fix_img_dir, fix_img_thumb_dir,
    downsample_factor=(1, 1, 1)
):
    mov_img_size = get_tif_size(mov_img_dir / '000.tif')
    mov_img_thumb_size = get_tif_size(mov_img_thumb_dir / '000.tif')
    fix_img_size = get_tif_size(fix_img_dir / '000.tif')
    fix_img_thumb_size = get_tif_size(fix_img_thumb_dir / '000.tif')
    print('Moving image size:', mov_img_size)
    print('Moving image thumbnail size:', mov_img_thumb_size)
    print('Fixed image size:', fix_img_size)
    print('Fixed image thumbnail size:', fix_img_thumb_size)

    print('Loading moving image stack')
    mov_img_stack = load_image_stack(mov_img_thumb_dir)
    print('Loading fixed image stack')
    fix_img_stack = load_image_stack(fix_img_thumb_dir)

    print('Normalizing image intensity')
    mov_img_stack = normalize_intensity(mov_img_stack)
    fix_img_stack = normalize_intensity(fix_img_stack)

    print('Padding image')
    mov_img_stack, fix_img_stack = pad([mov_img_stack, fix_img_stack])

    print(f'Downsampling image with factor {downsample_factor}')
    dx, dy, dz = downsample_factor
    mov_img_stack = mov_img_stack[::dx, ::dy, ::dz]
    fix_img_stack = fix_img_stack[::dx, ::dy, ::dz]

    print('Converting to AirLab image')
    dtype = torch.float32
    device = torch.device('cpu')
    origin = (0, 0, 0)
    downsample_factor = np.array(downsample_factor)

    def prep_al_img(img, size, thumb_size):
        spacing = np.append(size / thumb_size, 1) * downsample_factor
        img = torch.tensor(img, dtype=dtype).to(device)
        img = al.Image(img, img.shape, spacing, origin)
        return img

    mov_img = prep_al_img(mov_img_stack, mov_img_size, mov_img_thumb_size)
    fix_img = prep_al_img(fix_img_stack, fix_img_size, fix_img_thumb_size)
    return mov_img, fix_img

def get_tif_size(tif_file):
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(tif_file.as_posix())
    file_reader.ReadImageInformation()
    return np.array(file_reader.GetSize())

def load_image_stack(image_dir):
    image_dir = Path(image_dir).resolve()
    image_stack = []
    for image_file in sorted(image_dir.iterdir()):
        print(f'Loading image {image_file.name}', end='\r')
        image = sitk.ReadImage(image_file.as_posix())
        image_arr = sitk.GetArrayViewFromImage(image).copy()
        image_stack.append(image_arr.T)
    print(f'Finished loading {len(image_stack)} images')
    image_stack = np.stack(image_stack, axis=-1)
    return image_stack

def normalize_intensity(image):
    v_min, v_max = image.min(), image.max()
    image = (image - v_min) / (v_max - v_min)
    return image

def pad(images):
    shape = np.array(list(map(lambda image: image.shape, images))).max(axis=0)

    def put_in_corner(canvas, image):
        dx, dy, dz = image.shape
        canvas[0:dx,0:dy,0:dz] = image
        
    padded_images = []
    for image in images:
        canvas = np.zeros(shape, dtype=image.dtype)
        put_in_corner(canvas, image)
        padded_images.append(canvas)
    return padded_images

def affine_registrate(
    mov_img, fix_img,
    lr=None, niter=None,
    init_transform=None
):
    registration = al.PairwiseRegistration(verbose=True)

    transformation = al.transformation.pairwise.AffineTransformation(mov_img)
    if init_transform:
        init_transform.init_al_transform(transformation)
    registration.set_transformation(transformation)

    image_loss = al.loss.pairwise.MSE(fix_img, mov_img)
    registration.set_image_loss([image_loss])

    optimizer = torch.optim.Adam(transformation.parameters(), lr=lr)
    registration.set_optimizer(optimizer)

    registration.set_number_of_iterations(niter)
    registration.start()

    end = time.time()
    return transformation

def plot_diagnostic(brain, img_wrp, img_fix, work_dir, zstep=10):
    img_diff = img_wrp - img_fix
    with PdfPages(work_dir / f'diagnostic.pdf') as pdf:
        sz = img_fix.shape[-1]
        for z in range(0, sz, zstep):
            print(f'{z}/{sz}')
            px = 1 / plt.rcParams['figure.dpi']
            fig, ax = plt.subplots(1, 3, dpi=250, figsize=(8, 6))
            kwargs = {
                'aspect':'equal',
                'cmap': 'gray',
            }
            ax[0].imshow(img_wrp[:,:,z], **kwargs)
            ax[0].set_title('DK52 transformed')
            ax[0].set_axis_off()
            ax[1].imshow(img_fix[:,:,z], **kwargs)
            ax[1].set_title(f'{brain} fixed')
            ax[1].set_axis_off()
            kwargs['cmap'] = 'coolwarm'
            kwargs['vmin'] = -1
            kwargs['vmax'] = 1
            ax[2].imshow(img_diff[:,:,z], **kwargs)
            ax[2].set_title(f'DK52 (red) - {brain} (blue)')
            ax[2].set_axis_off()
            fig.suptitle(f'z = {z}')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

def plot_diagnostic_alt(brain, img_wrp, img_fix, work_dir, res, zstep=10):
    dx, dy, _ = res
    x = [10, 10 + 1000 / dx]
    y = [20, 20]
    with PdfPages(work_dir / f'diagnostic-alt.pdf') as pdf:
        sz = img_fix.shape[-1]
        for z in range(0, sz, zstep):
            print(f'{z}/{sz}')
            kwargs = {
                'aspect':'equal',
                'cmap': 'gray',
            }

            fig, ax = plt.subplots(1, 1, dpi=200, figsize=(8, 6))
            ax.imshow(img_wrp[:,:,z].T, **kwargs)
            ax.plot(x, y, '-w')
            ax.text(
                (x[0] + x[1]) / 2, y[0], '1 mm', c='white',
                horizontalalignment='center',
                verticalalignment='bottom'
            )
            ax.set_title(f'DK52 transformed\nz = {z}')
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(1, 1, dpi=200, figsize=(8, 6))
            ax.imshow(img_fix[:,:,z].T, **kwargs)
            ax.plot(x, y, '-w')
            ax.text(
                (x[0] + x[1]) / 2, y[0], '1 mm', c='white',
                horizontalalignment='center',
                verticalalignment='bottom'
            )
            ax.set_title(f'{brain} fixed\nz = {z}')
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('brain')
    parser.add_argument('--base_brain', default='DK52')
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--dx', help='downsampling factor in x',
        type=int, default=8)
    parser.add_argument('--dy', help='downsampling factor in x',
        type=int, default=8)
    parser.add_argument('--dz', help='downsampling factor in x',
        type=int, default=2)
    parser.add_argument('--lr', help='learning rate',
        type=float, default=1e-2)
    parser.add_argument('--niter', help='number of optimization iterations',
        type=int, default=64)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    data_dir = Path('/net/birdstore/Active_Atlas_Data/data_root/pipeline_data')
    work_dir = ROOT / 'data/rough-alignment' / args.brain
    work_dir.mkdir(parents=True, exist_ok=True)

    mov_brain = args.base_brain
    fix_brain = args.brain
    print(f'Aligning {mov_brain} (moving image) to {fix_brain} (fixed image)')
    if args.cont:
        print('Continue from last time')

    mov_img_dir = data_dir / mov_brain / 'preps/CH1/full_aligned'
    mov_img_thumb_dir = data_dir / mov_brain / 'preps/CH1/thumbnail_aligned'

    fix_img_dir = data_dir / fix_brain / 'preps/CH1/full_aligned'
    fix_img_thumb_dir = data_dir / fix_brain / 'preps/CH1/thumbnail_aligned'

    mov_img, fix_img = load_and_prep_images(
        mov_img_dir, mov_img_thumb_dir,
        fix_img_dir, fix_img_thumb_dir,
        downsample_factor=(args.dx, args.dy, args.dz)
    )

    transform_param_file = work_dir / f'transform-affine-al.json'
    init_transform = None
    if args.cont:
        init_transform = load_al_affine_transform(transform_param_file)
    transform = affine_registrate(
        mov_img, fix_img,
        lr=args.lr, niter=args.niter,
        init_transform=init_transform
    )

    if args.plot:
        print('Warping moving image')
        displacement = transform.get_displacement()
        wrp_img = al.transformation.utils.warp_image(mov_img, displacement)

    print('Saving results')

    dump_al_affine_transform(mov_img, fix_img, transform, transform_param_file)
    transform = load_al_affine_transform(transform_param_file)

    coms_file = work_dir / f'coms-rough.json'
    with open(ROOT / 'data/DK52_coms_kui_detected.json', 'r') as f:
        mov_coms = json.load(f)
    fix_coms = {}
    for name, com in mov_coms.items():
        com = np.array(com, dtype=float)
        fix_coms[name] = transform.forward_point(com).tolist()
    with open(coms_file, 'w') as f:
        json.dump(fix_coms, f, sort_keys=True, indent=4)

    if args.plot:
        print('Making plots')
        res = np.array([0.325, 0.325, 20]) * fix_img.spacing
        mov_img = mov_img.image[0,0].numpy()
        fix_img = fix_img.image[0,0].numpy()
        wrp_img = wrp_img.image[0,0].numpy()
        plot_diagnostic(args.brain, wrp_img, fix_img, work_dir)
        plot_diagnostic_alt(args.brain, wrp_img, fix_img, work_dir, res)

    print('Finished!\n')
