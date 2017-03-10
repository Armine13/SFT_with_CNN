Stony Brook University Real-world Clutter Dataset (SBU-RwC90)

- 90 images, 800x600 resolution. All sampled from the SUN09 dataset (http://groups.csail.mit.edu/vision/SUN/).
- 6 groups of 15 images each: group 1 has 1~10 objects, group 2 has 11~20 objects, up to group 6 that has 51~60 
objects.
- Object segmentations by human subjects for all 90 images are provided as part of SUN09.
- Clutter rankings done by 15 human subjects are provided, experiements conducted by SBU. Ground truth clutter 
rating of each image is its median ranked position by the human subjects. Mean correlation between all pairs of
human rankings = 0.6919 (Spearman's rho, p < 0.001).

Contact: cheyu@cs.stonybrook.edu

If you use our dataset, please cite the following paper:

Modeling Clutter Perception using Parametric Proto-Object Partitioning.
Chen-Ping Yu, Wen-Yu Hua, Dimitris Samaras, and Gregory Zelinsky
Advances in Neural Information Processing Systems (NIPS), Lake Tahoe USA, 2013 

bibtex:
@incollection{Yu_NIPS2013,
title = {Modeling Clutter Perception using Parametric Proto-object Partitioning},
author = {Chen-Ping Yu and Wen-Yu Hua and Dimitris Samaras and Gregory Zelinsky},
booktitle = {Advances in Neural Information Processing Systems 26},
pages = {118--126},
year = {2013},
}

Zip Content:
grouped_by_obj_count: the name of each sub-directory indicates the # of human segmented objects that each member-image contains. Each category
                      contains 15 images, that totals to 90 images.
mixed: the 90 images of entire dataset. Ordered by image ID (IDs are ordered alphabetically by the file names).
segmentations_mixed: contains the human object segmentation for all 90 images (named by image ID 1-90), obtained from SUN09.
human_ratings_median.mat: matlab file that contains a 90-element column vector. The rows are ordered by the image ID 1-90, and the value
                          is the image's median ranked position by the 15 human subjects. For example, image #3 has a median ranking of 17/90.
segmentation.xls: contains the information of all 90 images: the corresponding img ID and the image name, its SUN09 directory, and its associated
                  number of objects segmented by humans, provided by SUN09. 					  