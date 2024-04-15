"""
strokedecomposition.py

stroke decomposition algorithms take in a preprocessed image and return a list of extracted strokes
* strokes are approximations
"""

import numpy as np
import cv2 as cv
import qdanalysis.preprocessing as prep

from skimage.morphology import skeletonize
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import find_objects


#skeleton network library
import sknw

#get height and width of the slice object and pass (False) or fail (True)
def reject_slice(bb, im_shape):
    
    #sometimes scipy returns a None instead of a slice. not sure why, but can gloss over it for now
    if bb is None:
        return True
    
    h_slice, w_slice = bb
    #should always be positive, but abs is a guarantee
    height = abs(h_slice.stop - h_slice.start)
    width = abs(w_slice.stop - w_slice.start)

    #reject thin stroke regions
    thin_t = 3
    if min(height, width) <= thin_t:
        return True
    
    #reject small blobs (where both dimensions are less than blob_t)
    blob_t = 10
    if height <= blob_t and width <= blob_t:
        return True
    
    #reject overly long shapes
    im_height, im_width = im_shape
    height_perc = 0.75
    width_perc = 0.75
    if height > im_height*height_perc or width > im_width*width_perc:
        return True
    
    return False

    

"""
uses KNearestNeighbors to take a label image and a foreground image and group the non-zero pixels of that foreground image according to the closest pixel label.
* labels and foreground should be of the same size
* labels is a integer array of n different class labels
"""
def knnRegionGrowth(labels, foreground):
    #this is the "train" and "test" set for the knn classifier, what the other values are going to be matched to
    label_coords = np.transpose(labels.nonzero())
    label_vals = labels[label_coords[:, 0], label_coords[:, 1]]
    
    #knn classifier will label foreground element via closest skeleton point
    cls = KNeighborsClassifier(n_neighbors=3)
    cls.fit(label_coords, label_vals)

    #now grab the coordinates of all the foreground elements and match them to a label
    img_coords = np.transpose(foreground.nonzero())
    img_labels = cls.predict(img_coords)

    segmented_image = np.zeros_like(foreground, dtype=int)
    segmented_image[img_coords[:, 0], img_coords[:, 1]] = img_labels

    return segmented_image

#assigns an integer label, 1, 2, .., n for
def label_graph_edges(im_graph, im_shape):
    #array, the size of the image that the labels are written onto
    labels = np.zeros(shape=im_shape)

    #construct a labeled image from the graph consisting of all edges
    for label_idx, (node1, node2, idx) in enumerate(im_graph.edges):
        edge_points = im_graph[node1][node2][idx]['pts']
        labels[edge_points[:, 0], edge_points[:, 1]] = label_idx + 1 #need to account for zero indexing

    return labels

#assigns an integer label 1, 2, .., n for both the nodes and edges
def label_graph_edges_and_nodes(im_graph, im_shape):
    #array, the size of the image that the labels are written onto
    label_img = np.zeros(shape=im_shape)
    #labels from 1, 2, .. n (0 is background)
    label_idx = 1
    label_attr = 'label'
    
    for node_id in im_graph.nodes:
        node_points = im_graph.nodes[node_id]['pts']
        label_img[node_points[:, 0], node_points[:, 1]] = label_idx

        #assign label attr to node
        im_graph.nodes[node_id][label_attr] = label_idx
        label_idx += 1

    #construct a labeled image from the graph consisting of all edges
    for (node1, node2, idx) in im_graph.edges:
        edge_points = im_graph[node1][node2][idx]['pts']

        label_img[edge_points[:, 0], edge_points[:, 1]] = label_idx #need to account for zero indexing

        im_graph[node1][node2][idx][label_attr] = label_idx
        label_idx += 1

    #NOTE: maybe assign as graph attribute?
    return label_img

"""
baseline stroke decomposition for graph based techniques. simply skeletonizes an image and turns it into a graph to 
    extract edge segments. implicitly returns grayscale image if passed

parameters:
* image - a preprocessed image represented as an nd_array
"""
def simple_stroke_segment(image):

    #step 1
    image_gs, image_bin = prep.preprocess(image)

    #step 2
    # uses Zhang's algorithm as default for 2D
    im_skeleton = skeletonize(image_bin)

    #step 3
    #convert to skeleton to graph representation
    #Full doesn't seem to be properly implemented. seems to just set the 'start' and 'end' pixel coordinates to the mean of the node pixels
    im_graph = sknw.build_sknw(im_skeleton, multi=True, full=False, ring=True)

    #step 4
    labels = label_graph_edges(im_graph, image_bin.shape)
    #step 5
    #region growing
    region_attr = 'regions'
    im_graph.graph[region_attr] = knnRegionGrowth(labels, image_bin)

    #step 6 and 7
    #bounding box coords of image labels, should line up with label numbers
    stroke_bb = find_objects(im_graph.graph[region_attr])

    #now that we have segmented the image, we need to extract the segments as a list of individual strokes
    extracted_strokes = []
    for idx, bb in enumerate(stroke_bb):
        
        #skip slice if fails the filter_criteron (step 6)
        if reject_slice(bb, image_gs.shape):
            continue

        #get bounding box of segmented label and filter any other labels in that bounding box
        filter = (im_graph.graph[region_attr][bb] == idx + 1)
        masked_gs = image_gs[bb] * filter.astype(int)
        extracted_strokes.append(masked_gs)

    return extracted_strokes



