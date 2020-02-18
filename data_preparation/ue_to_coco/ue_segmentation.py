import csv
import math
import cv2
import numpy as np
import glob
import os

metadata_folder = os.path.join("ue_to_coco", "Metadata")
masks_folder = os.path.join("ue_to_coco", "Masks")
metadata_name = "Segmentation.csv"
metadata_delimiter = "-"
segmentation_name_ext = "_s"
img_extension = ".png"


def get_colours():
    """
    Open the CSV file created by UE and extract the available data

    :return dict_colours: dictionary with colours, tags and object ids
    :rtype: dictionary
        key: coulour tuple (R,G,B) as int
        value: tuple of object id and category ('tag' in UE)
    """
    dict_colours = {}

    # open metadata file and save structure to dict_colours
    with open(os.path.join(metadata_folder, metadata_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=metadata_delimiter)
        line_count = 0
        for row in csv_reader:
            colour = row[2][1:-1].split(",")
            # float colour to int for each channel (and remove alpha)
            colour = tuple([int(math.ceil(float(channel[2:])*255)) for channel in colour][:-1])
            # key = colour; values = id, category
            dict_colours[colour] = (row[0], row[1])
            line_count += 1
        print(f'Processed {line_count} objects for segmentation.')
    return dict_colours


def get_polygons(image_name, dict_colours):
    """
    Process flat colour images from UE and associated metadata
    and return polygon points and bounding boxes

    :param str image_name: path to the image
    :param dictionary dict_colours: colour dictionary
    :return list_poly: list of polygon points for each object on the image as well as
        it's bounding box, object id, category and contour total area, if the the object is
        split into multiple areas, each of them has a new list with polygon points
    :rtype: list
        structure: [[(x1, y1), (x2, y2),...], [(x1, y1), (x2, y2),...], (bbox), (id, category, area)]
    """
    list_poly = []

    # load the flat coloured mask image
    image_base_name, image_ext = os.path.splitext(os.path.basename(image_name))
    mask_name = os.path.join(masks_folder, image_base_name + segmentation_name_ext + image_ext)
    mask = cv2.imread(mask_name)

    # process all colours to find the contours and bounding boxes
    for colour in dict_colours.keys():
        # define boundaries
        # turn RGB into BGR (since OpenCV represents images as NumPy arrays in reverse order)
        R, G, B = colour
        upper = [B, G, R]
        # upper = [x if x+1>255 else x+1 for x in upper]
        lower = [x-2 for x in upper]

        # convert boundaries to NumPy arrays
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colours within the specified boundaries and apply the mask
        object_mask = cv2.inRange(mask, lower, upper)
        # isolated_object = cv2.bitwise_and(image_flat, image_flat, mask = object_mask)
        # cv2.imshow('segmented_image', mask)
        # cv2.waitKey()

        # get contours using point approximation
        countours = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        countours = countours[0] if len(countours) == 2 else countours[1]

        # add countours to segmentation structure
        bbox_list = []
        list_contours = []
        contour_area = 0
        for contour in countours:
            # get bounding box
            bbox_list.append(cv2.boundingRect(contour))
            # get total area
            contour_area += cv2.contourArea(contour)
            # get contour poins
            contour_point_list = []
            for i in contour:
                for j in i:
                    contour_point_list.append((int(j[0]), int(j[1])))

            if contour_point_list:
                list_contours.append(contour_point_list)

        # convert per polygon bounding boxes to all encompasing bounding box
        x_min, y_min = 100000, 100000
        x_max, y_max = 0, 0
        if bbox_list:
            for bbox in bbox_list:
                if bbox[0] < x_min:
                    x_min = bbox[0]
                if bbox[1] < y_min:
                    y_min = bbox[1]
                if bbox[0] + bbox[2] > x_max:
                    x_max = bbox[0] + bbox[2]
                if bbox[1] + bbox[3] > y_max:
                    y_max = bbox[1] + bbox[3]

            bbox_full = (x_min, y_min, x_max - x_min, y_max - y_min)  # X0, Y0, Xsize, Ysize
            list_contours.append(bbox_full)
            list_contours.append(dict_colours[colour] + (contour_area,))
            list_poly.append(list_contours)

    return list_poly


def get_segmentation(images_folder):
    """
    Get segmentation data for all images in a folder

    :param string images_folder: path to the folder of images to process (ex. /data/train/)
    :return dict_segmentation: dictionary of polygon points for each object on the image as well as
        it's bounding box, object id, category and contour total area, if the the object is
        split into multiple areas, each of them has a new list with polygon points
    :rtype: dictionary
        key: image path
        value: [[(x1, y1), (x2, y2),...], [(x1, y1), (x2, y2),...], (bbox), (id, category, area)]
    """
    dict_segmentation = {}
    # open csv file from UE and retrieve the data
    # key = colour; values = id, category
    dict_colours = get_colours()

    # get all image file names excluding the the segmentation ones ending wih "_s"
    images = [f for f in glob.glob(os.path.join(images_folder, f"*[!{segmentation_name_ext}]" + img_extension))]

    for image in images:
        list_poly = get_polygons(image, dict_colours)
        if list_poly:
            dict_segmentation[image] = list_poly

    return dict_segmentation


def debug_draw(dict_segmentation, image_index):
    """
    Draw the segmentation on to an image for debug

    :param dictionary dict_segmentation: segmentation
    :param int image_index: specific image index from the collection
    """
    # load the images
    image_name, segmentation = list(dict_segmentation.items())[image_index]
    image_orig = cv2.imread(image_name)

    for data in segmentation:
        # draw contour
        # cv2.drawContours(image_orig, np.array(contour, dtype = "int"), 0, (0, 255, 255), 2)
        # draw bounding box
        bb = data[-2]
        cv2.rectangle(image_orig, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255, 0, 0), 1)
        # draw contour poins
        for contour in data[:-2]:
            for point in contour:
                cv2.circle(image_orig, point, 1, (0, 0, 255), 2)

    cv2.imshow('segmented_image', image_orig)
    cv2.waitKey()


def main():
    debug_draw(get_segmentation(os.path.join("ue_to_coco", "Images", "val")), 0)


if __name__ == "__main__":
    # execute only if run as a script
    main()
