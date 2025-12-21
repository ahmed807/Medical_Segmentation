config.py

Acts as a central settings file.

It tells the script where to find your raw_data and where to put the new coco_dataset. This makes it easy to change paths later without digging into the main logic.

coco_helpers.py

This is a "toolbox" of reusable functions.

init_coco_structure: Creates the empty "skeleton" (the basic dictionary) for the _annotations.coco.json file.

get_or_create_category: Manages the "categories" list in your JSON. It checks if a class (like "dog" or "bicycle") already exists. If not, it adds it and assigns it a new ID. This is also where we added the temporary fix to correct "horse" to "dog".

process_mask: This is the core-work function. It opens one of your black and white annotation_segmap_...png files, traces the outline of the white shape (using cv2.findContours), and calculates the three key pieces of information COCO needs:

segmentation: A list of [x, y] coordinates for the object's outline.

area: The total number of pixels in the object.

bbox: The [x, y, width, height] bounding box that encloses the object.

convert.py

This is the main script that you run. It acts as the "manager."

It imports the settings from config.py and the tools from coco_helpers.py.

It loops through each folder (0, 1, 2, etc.) inside your raw_data directory.

For each folder, it finds the original.png and labels.json.

It reads the original.png and saves a new .jpg version of it in the coco_dataset/train folder.

It reads the labels.json to find out which mask files to look for (e.g., _0_0, _0_1).

For each object, it calls the helper functions to get the category ID and process the mask.

It assembles all this information into the final COCO dictionary.

Finally, after processing all folders, it saves everything into the single _annotations.coco.json file.