# YOLO Label Assignment Exploration Repository

Welcome to this repository, where I’ll be diving deep into the fascinating world of YOLO! Here, I aim to explore various aspects of the YOLO model. *Your feedback gives me a lot of motivation to keep exploring different facets of YOLO*—whether it's through creating issues, sending emails, or giving this repository a star if you find the content valuable. I'll start by discussing the label assignment problem in the YOLO series.

## Introduction

A core idea behind the YOLO series is how object detection happens in the final layer (or layers). How does a deep neural network output bounding boxes for all the objects in an image, along with their predicted classes? And with images containing a variable number of objects, how can the model flexibly handle and output different numbers of bounding boxes? Take a moment to consider how you might solve this before we dive deeper.

When an image is processed by a YOLO network, the output is a feature map structured as S × S × Depth. Each grid cell in the feature map corresponds to a specific region of the input image. For instance, the top left of the feature map corresponds to the top left of the image. I'll refer to each grid cell (Width × Height) as a "Detection Unit (DU)," with Depth representing the unit's depth.

Each detection unit in YOLO has a specific job: it either outputs the coordinates for a bounding box of an object or signals that no object was detected. Take YOLOv1 as an example, where the output consists of a 7x7 grid. So, there are 49 DUs in total. The key question is: If an image has two objects, which of these DUs is responsible for detecting the two objects?

To address this, the authors of YOLOv1 introduced what I call the idea of "local responsibility". They propose that a DU is responsible for detecting an object's bounding box if the object's center falls within the region covered by that DU in the overlaid grid. If no object center maps back to a particular DU, then that DU is not responsible for detection of any objects for that image.

![Detection Unit](./assets/yolo-du.png)

Ground truth (GT) bounding boxes serve as labels, which must be assigned to specific DUs in a process known as "Label Assignment." Efficient label assignment is crucial, as it directly impacts how we define the loss function and begin the training process. In the previous paragraph, I touched on how YOLOv1 tackles this issue, and in the next section, I’ll delve deeper into Label Assignment and explore the innovative techniques introduced in different YOLO versions to make this process more efficient.

## YOLOv1

In YOLOv1, each Detection Unit has the following structure: x1,y1,width1,height1,objectnessscore1,x2,y2,width2,height2,objectnessscore2,classAprobability,classBprobability,…x1,y1,width1,height1,objectnessscore1,x2,y2,width2,height2,objectnessscore2,classA probability,classB probability,….  
This means each DU outputs two sets of bounding box coordinates along with corresponding objectness scores, and a set of class probabilities. For a dataset with 80 different classes, the total length of a DU would be 90: five values for the first bounding box, five for the second, and 80 class probabilities. Since YOLOv1 operates with a 7x7 grid, there are 49 DUs overall.

During training, for each image in a batch, we need to assign the ground truth bounding boxes to the appropriate Detection Units. Each DU can predict two bounding boxes. If the center of an object falls within a DU, that unit is responsible for detecting the object. But which of the two bounding boxes should be assigned to the ground truth? The answer is the one with the highest Intersection over Union (IoU) with the ground truth bounding box. The Objectness score of this bounding box is pushed toward 1, indicating a positive detection. The other bounding box, along with all other DUs that do not correspond to any object center, will have their Objectness scores pushed toward 0, indicating no object detected.

The reason YOLOv1 predicts two bounding boxes per DU, as explained in the video linked here, is that the model hopes one bounding box will specialize in detecting wide objects, while the other will specialize in detecting tall objects.

## YOLOv2, YOLOv3, PPYOLO, and PPYOLOv2

In YOLOv2, YOLOv3, PPYOLO, and PPYOLOv2, each DU can predict multiple bounding boxes. While the specific details of these predictions are not relevant to this section, the key idea is that when a GT center falls within a DU, the predicted bounding box with the highest IoU is assigned to that GT.

## YOLOv5

YOLOv5 introduces some modifications compared to previous versions. First, it retains the idea of assigning multiple bounding boxes in a DU to a GT object if the center of the object falls within that DU. However, instead of relying on Intersection over Union (IoU) as the matching criterion, YOLOv5 adopts a different measurement. It considers all anchor boxes whose maximum width-height ratio to the GT box is below a certain threshold as a match to the GT. In simple terms, YOLOv5 checks how much the predicted box needs to stretch or shrink in both width and height to fit the actual object. If the largest change is small enough (below a given limit), the predicted box is matched to the object.

More specifically, the ratio of the GT box height to the predicted box height is calculated, along with its reciprocal, and the maximum of the two is taken. The same process is done for the width. This gives us the largest change needed for either the GT or predicted box to adjust in width or height. Then, we calculate the maximum of these two values, meaning the maximum adjustment needed for either the width or height of the prediction. If this maximum adjustment is less than a certain threshold, a hyperparameter, the prediction is assigned to the GT.

There is also one more change: up to YOLOv4, if the center of an object fell into a DU, that DU would be solely responsible for detecting the object. In YOLOv5, this local responsibility extends to the neighboring non-diagonal DUs as well. So, if an object’s center falls into a specific part of a DU, like the top-left, not only that DU but also the immediate upper DU and immediate left DU will share responsibility for detecting the object.

[Documentation on YOLOv5](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/#44-build-targets).

## FCOS

While we've explored several YOLO versions, it's important to discuss another key object detection method, FCOS, as its ideas are incorporated into later YOLO versions.

In the YOLO versions we discussed so far, only the DU corresponding to the object's center was used, ignoring other DUs whose mapping fell within the object’s bounding box but weren't at the center. For example, in this image, DU3 corresponds to the object's center, while other DUs that are mapped to some other part of the bounding box were unused. In FCOS, however, all DUs (including DU1, DU2, and DU3) within the bounding box are utilized. Instead of just one DU, every DU that falls within the bounding box is assigned to that GT. Unlike early YOLO versions, FCOS leverages all points within the ground truth bounding box to predict bounding boxes (while low-quality predictions are suppressed using the 'center-ness' branch).

![FCOS Detection](./assets/yolo-fcos.png)

...

## Conclusion

It's fascinating to see how the label assignment problem has evolved over time. In YOLOv1-4, only the DU that lands on the center of a GT bounding box is assigned to that GT. YOLOv5 relaxes this a bit, extending responsibility to the DUs immediately surrounding the center as well. FCOS goes further by assigning responsibility to all DUs that fall within a GT’s bounding box.

YOLOX introduces a more balanced approach by assigning responsibility not only to the center DU but also to its neighboring DUs, while also considering a global view that minimizes the overall loss across all DU-GT pairs. TOOD builds on this idea by defining a score that accounts for both classification and regression simultaneously, similar to YOLOX in some ways.
