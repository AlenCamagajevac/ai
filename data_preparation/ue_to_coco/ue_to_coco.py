from datetime import datetime
import ue_segmentation as ue
import os
import imagesize
import json

# file save name
json_name_base = "_coco.json"

image_folders = [os.path.join("ue_to_coco", "Images", "train"),
                 os.path.join("ue_to_coco", "Images", "val")]

# structure of the json
json_dict = {
    "info":
    {
        "description": "UE Dataset",
        "url": "https://protostar.ai/",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "Protostar Labs",
        "date_created": datetime.today().strftime("%Y/%m/%d")
    },
    "licenses": [
        {
            "url": "???",
            "id": 1,
            "name": "???"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": []
}


def make_categories(ue_dict):
    """
    Create categories from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_categories: list of dictionaries containing categories
    :rtype: list
        structure: [{supercategory, id, name}, {},...]
    """
    list_categories = []
    list_cat_appeared = []
    counter_cat = 0
    for data in ue_dict.values():
        for contours in data:
            category = contours[-1][1]
            if category not in list_cat_appeared:
                dict_category = {}
                list_cat_appeared.append(category)
                dict_category["supercategory"] = "object"
                dict_category["id"] = counter_cat
                dict_category["name"] = category
                counter_cat += 1

                list_categories.append(dict_category)

    return list_categories


def make_images(ue_dict):
    """
    Create image data from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_images: list of dictionaries containing image data
    :rtype: list
        structure: [{}, {},...]
    """
    list_images = []
    for image_id, image in enumerate(ue_dict.keys()):
        dict_image = {}
        modified_date = datetime.utcfromtimestamp(os.path.getmtime(image)).strftime('%Y-%m-%d %H:%M:%S')
        # width, height = get_image_size.get_image_size(image)
        width, height = imagesize.get(image)
        dict_image["licence"] = 1
        dict_image["file_name"] = os.path.basename(image)
        dict_image["coco_url"] = ""
        dict_image["height"] = height
        dict_image["width"] = width
        dict_image["date_captured"] = modified_date
        dict_image["flickr_url"] = ""
        dict_image["id"] = image_id + 1

        list_images.append(dict_image)

    return list_images


def make_annotations(ue_dict):
    """
    Create annotation data from processed UE data

    :param dictionary ue_dict: segmentation data
    :return list_annotations: list of dictionaries containing annotation data
    :rtype: list
        structure: [{}, {},...]
    """
    list_annotations = []
    ann_counter = 1
    for image_id, objects in enumerate(ue_dict.values()):
        for contours in objects:
            dict_contours = {}
            dict_contours["segmentation"] = []
            # convert a list of tuples of ints into a list of ints
            for cont_list in contours[:-2]:
                dict_contours["segmentation"].append([coord for pair in cont_list for coord in pair])
            dict_contours["area"] = contours[-1][2]
            dict_contours["iscrowd"] = 0
            dict_contours["image_id"] = image_id + 1
            dict_contours["bbox"] = list(contours[-2])
            dict_contours["category_id"] = next((item.get("id") for item in json_dict["categories"] if item["name"] ==
                                                contours[-1][1]), 0)
            dict_contours["id"] = ann_counter  # contours[-1][0]
            ann_counter += 1

            list_annotations.append(dict_contours)

    return list_annotations


def main():
    """
    Create json in COCO format for all images
    """

    for path in image_folders:
        ue_dict = ue.get_segmentation(path)
        json_dict["categories"] = make_categories(ue_dict)
        json_dict["images"] = make_images(ue_dict)
        json_dict["annotations"] = make_annotations(ue_dict)

        # print(json.dumps(json_dict))
        json_name = os.path.basename(path) + json_name_base
        with open(os.path.join(ue.metadata_folder, json_name), 'w') as outfile:
            json.dump(json_dict, outfile)

        print(f"Exported segmentation json file in COCO format to {os.path.join(ue.metadata_folder, json_name)}")


main()
