This is the Gallbladder Cancer Ultrasound (GBCU) Dataset, released with the IEEE/CVF CVPR 2022 paper titled "Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG with Curriculum Learning".

Note, all bounding box coordinates are of the format, (x_min, y_min, x_max, y_max).

1. "imgs/": contains the actual .jpg images. Image names are in "imXXXXX.jpg" format.

2. "bbox_annot.json": contains the bounding box annotations drawn by the radiologists. Each image has a large bounding box with label 'nml' or 'abn', to localize the Gallbladder and reltaed region of interest (ROI). The images may contain additional bounding boxes for pathologies, such as stone, malignancy, bening mural thickening, etc. However, these bounding boxes were not used in the original work.

3. "roi_pred.json": contains the images with the coordinates of the bounding box predicted by the region selection network. The ground-truth (gold) bounding box coordinates, and IoU is also provided.

4. "train.txt"/ "test.txt": contain the labels for each image used for classification. 0 - normal, 1 - benign, and 2 - malignant.
