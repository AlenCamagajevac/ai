from cv2 import cv2
import glob
import os

images_folder = os.path.join("data_preparation", "ue_to_unet", "Images")
masks_folder = os.path.join("data_preparation", "ue_to_unet", "Masks")
masks_binary_folder = os.path.join("data_preparation", "ue_to_unet", "MasksBinary")
segmentation_name_ext = "_s"


def make_mask_binary(image_path):
    """
    Process flat colour images from UE and return their binary version (black or white)

    :param str image_path: path to the image
    :return mask_binary, mask_name: binary version of the segmentation image and it's name
    :rtype: cv2 image array, string
    """

    # load the flat coloured mask image and convert to binary image
    mask_name = image_path.split(".")[1].split("\\")[-1] + segmentation_name_ext + "." + image_path.split(".")[2]
    mask_name_path = masks_folder + mask_name
    mask = cv2.imread(mask_name_path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

    # DEBUG
    # cv2.imshow("Binary", mask_binary)
    # cv2.waitKey(0)

    return mask_binary, mask_name


def get_segmentation():
    """
    Get segmentation data for all images in a folder and save the binary segmentation version in a diferent folder

    :return: none
    """

    # get all image file names excluding the the segmentation ones ending wih "_s"
    images = [f for f in glob.glob(images_folder + f"*[!{segmentation_name_ext}].png")]

    for image_path in images:
        image_mask_binary, mask_name = make_mask_binary(image_path)
        # Saving the image
        cv2.imwrite(masks_binary_folder + mask_name, image_mask_binary)

    print(f'Processed {len(images)} UE images for U-Net.')


def main():
    get_segmentation()


main()
