"""This module is responsible for creating and applying the masks. The idea
and lots of the code were very heavily borrowed from this study:
https://www.cis.upenn.edu/~jshi/ped_html/
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from skimage import color
Image.MAX_IMAGE_PIXELS = None
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from library.database_model.scan_run import BOTTOM_MASK
from library.utilities.utilities_mask import combine_dims, merge_mask
from library.utilities.utilities_process import read_image, test_dir, get_image_size, write_image


class MaskManager:
    """Class containing all methods related to image masks
    This class is used to create masks for each tiff image (post-extraction from czi files), 
    apply user-modified masks

    Note: Uses pytorch for ML generation of masks
    """

    def apply_user_mask_edits(self):
        """Apply the edits made on the image masks to extract the tissue from the 
        surround debris to create the final masks used to clean the images.
        Input dir is the colored merged masks
        """
        
        self.input = self.fileLocationManager.get_thumbnail_colored(self.channel)
        self.output = self.fileLocationManager.get_thumbnail_masked(self.channel)
        
        test_dir(self.animal, self.input, self.section_count, True, same_size=False)
        os.makedirs(self.output, exist_ok=True)
        files = sorted(os.listdir(self.input))
        self.logevent(f"Input FOLDER: {self.input}")
        self.logevent(f"FILE COUNT: {len(files)}")
        self.logevent(f"MASKS FOLDER: {self.output}")
        for file in files:
            filepath = os.path.join(self.input, file)
            maskpath = os.path.join(self.output, file)
            if os.path.exists(maskpath):
                continue
            mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if self.mask_image > 0:
                mask = mask[:, :, 2]
                mask[mask > 0] = 255

            cv2.imwrite(maskpath, mask.astype(np.uint8))

        if self.mask_image == BOTTOM_MASK:
            for file in files:
                maskpath = os.path.join(self.output, file)
                maskfillpath = os.path.join(self.output, file)   
                mask = read_image(maskfillpath)
                white = np.where(mask==255)
                whiterows = white[0]
                #whitecols = white[1]
                firstrow = whiterows[0]
                lastrow = whiterows[-1]
                lastcol = max(white[1])
                mask[firstrow:lastrow, 0:lastcol] = 255
                write_image(maskfillpath, mask.astype(np.uint8))


    def get_model_instance_segmentation(self, num_classes):
        """This loads the mask model CNN

        :param num_classes: int showing how many classes, usually 2, brain tissue, not brain tissue
        """

        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        return model

    def create_mask(self):
        """Helper method to call either full resolition of downsampled.
        Create the images masks for extracting the tissue from the surrounding 
        debris using a CNN based machine learning algorithm.
        If the images are from the MDXXX brains, they are 3 dimenions so the masks
        need to be done differently
        """
        
        if self.channel == 1:
            if self.downsample:
                self.create_downsampled_mask()
            else:
                self.create_full_resolution_mask()

    def load_machine_learning_model(self):
        """Load the CNN model used to generate image masks
        """
        
        modelpath = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/mask.model.pth")
        self.loaded_model = self.get_model_instance_segmentation(num_classes=2)
        workers = 2
        batch_size = 4
        torch.multiprocessing.set_sharing_strategy('file_system')

        device = torch.device('cpu')
        print(f' using CPU with {workers} workers at a batch size of {batch_size}')

        if os.path.exists(modelpath):
            self.loaded_model.load_state_dict(torch.load(modelpath, map_location = device, weights_only=False))
        else:
            print("no model to load")
            return

    def create_full_resolution_mask(self):
        """Upsample the masks created for the downsampled images to the full resolution
        """
        
        self.input = self.fileLocationManager.get_full(self.channel)
        THUMBNAIL = self.fileLocationManager.get_thumbnail_masked(channel=self.channel) # usually channel=1, except for step 6
        self.output = self.fileLocationManager.get_full_masked(channel=self.channel) # usually channel=1, except for step 6
        self.logevent(f"Input FOLDER: {self.input}")
        starting_files = os.listdir(self.input)
        self.logevent(f"FILE COUNT: {len(starting_files)}")
        self.logevent(f"Output FOLDER: {self.output}")
        test_dir(
            self.animal, self.input, self.section_count, self.downsample, same_size=False
        )
        os.makedirs(self.output, exist_ok=True)
        files = sorted(os.listdir(self.input))
        file_keys = []
        for file in tqdm(files):
            infile = os.path.join(self.input, file)
            thumbfile = os.path.join(THUMBNAIL, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue
            try:
                width, height = get_image_size(infile)
            except:
                print(f"Could not open {infile}")
            size = int(width), int(height)
            file_keys.append([thumbfile, outfile, size])

        workers = self.get_nworkers()
        self.run_commands_concurrently(self.resize_tif, file_keys, workers)

    def create_contour_mask(self):
        """Create masks for the downsampled images using contours
        The input files are the files that have not been normalized
        The output files are the colored merged files. 
        """
        
        self.input = self.fileLocationManager.get_thumbnail(self.channel)
        self.output = self.fileLocationManager.get_thumbnail_masked(channel=1)
        os.makedirs(self.output, exist_ok=True)
        
        test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)
        files = os.listdir(self.input)
        for file in files:
            infile = os.path.join(self.input, file)
            mask_dest_file = (os.path.splitext(file)[0] + ".tif")
            maskpath = os.path.join(self.output, mask_dest_file)

            if os.path.exists(maskpath):
                continue

            img = read_image(infile)
            new_img = color.rgb2gray(img)
            new_img *= 255 # or any coefficient
            new_img = new_img.astype(np.uint8)
            new_img[(new_img > 200)] = 0
            lowerbound = 0
            upperbound = 255
            #all pixels value above lowerbound will  be set to upperbound 
            _, thresh = cv2.threshold(new_img.copy(), lowerbound, upperbound, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
            thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
            mask = cv2.bitwise_not(thresh)            
            cv2.imwrite(maskpath, mask.astype(np.uint8))

    def create_downsampled_mask(self):
        """Create masks for the downsampled images using a machine learning algorithm.
        The input files are the files that have been normalized.
        The output files are the colored merged files. 
        """
        
        self.load_machine_learning_model()
        transform = torchvision.transforms.ToTensor()
        self.input = self.fileLocationManager.get_normalized(self.channel)
        self.output = self.fileLocationManager.get_thumbnail_colored(channel=self.channel) # usually channel=1, except for step 6
        self.logevent(f"Input FOLDER: {self.input}")
        
        test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)
        os.makedirs(self.output, exist_ok=True)
        files = os.listdir(self.input)
        self.logevent(f"FILE COUNT: {len(files)}")
        self.logevent(f"self.output FOLDER: {self.output}")
        for file in tqdm(files):
            filepath = os.path.join(self.input, file)
            mask_dest_file = (os.path.splitext(file)[0] + ".tif")
            maskpath = os.path.join(self.output, mask_dest_file)

            if os.path.exists(maskpath):
                continue

            img = Image.open(filepath)
            if self.mask_image > 0:
                torch_input = transform(img)
                torch_input = torch_input.unsqueeze(0)
                self.loaded_model.eval()
                with torch.no_grad():
                    pred = self.loaded_model(torch_input)
                masks = [(pred[0]["masks"] > 0.25).squeeze().detach().cpu().numpy()]
                mask = masks[0]
                dims = mask.ndim
                if dims > 2:
                    mask = combine_dims(mask)
                raw_img = np.array(img)
                mask = mask.astype(np.uint8)
                mask[mask > 0] = 255
                merged_img = merge_mask(raw_img, mask)
                del mask
            else:
                img = np.array(img)
                merged_img = np.zeros(img.shape)
                merged_img = merged_img.astype(np.uint8)
                merged_img[merged_img == 0] = 255
            cv2.imwrite(maskpath, merged_img.astype(np.uint8))

    @staticmethod
    def resize_tif(file_key):
        """Function to upsample mask images

        :param file_key: tuple of inputs to the upsampling program including:
        
        - path to thumbnail file
        - The output directory of upsampled image
        - resulting size after upsampling
        """

        thumbfile, outpath, size = file_key
        try:
            im = Image.open(thumbfile)
            im = im.resize(size, Image.LANCZOS)
            im.save(outpath)
        except IOError:
            print("cannot resize", thumbfile)
