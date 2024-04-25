# #%% Using mtcnn
# import matplotlib.pyplot as plt
# from mtcnn.mtcnn import MTCNN
# from matplotlib.patches import Rectangle
# from matplotlib.patches import Circle
# import cv2
# import os


# def draw_facebox(filename, result_list,output_fname):
#     # load the image
#     data = plt.imread(filename)
#     # plot the image
#     plt.imshow(data)
#     data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB )
#     # get the context for drawing boxes
#     ax = plt.gca()
#     # plot each box
#     print(result_list)
#     for i,result in enumerate(result_list):
#         # get coordinates
#         x, y, width, height = result['box']
#         # create the shape
#         cv2.imwrite(f"dataset\{output_fname}1.png",data[y:y+height,x:x+width])
#         rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
#         # draw the box
#         ax.add_patch(rect)
#         # show the plot
#     plt.show()# filename = 'test1.jpg' # filename is defined above, otherwise uncomment
#     # load image from file

# filename = 'dataset\\ankith1.jpg'
# pixels = plt.imread(filename) # defined above, otherwise uncomment
# # detector is defined above, otherwise uncomment
# print(pixels.shape)
# detector = MTCNN()
# # detect faces in the image
# faces = detector.detect_faces(pixels)
# # output filename
# out_name=f'ankith'
# # display faces on the original image
# draw_facebox(filename, faces,out_name)

# %% Using MediaPipe

from typing import Tuple, Union
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
    image,
    detection_result,Drawkeypoints=False):
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape
    cropped_list = []
    startpts_list=[]

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
        cropped = image[bbox.origin_y:bbox.origin_y+bbox.height,bbox.origin_x:bbox.origin_x+bbox.width]
        cropped_list.append(cropped)
        startpts_list.append(start_point)
        # Draw keypoints
        if Drawkeypoints:
            for keypoint in detection.keypoints:
                keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                                width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = '' if category_name is None else category_name
                probability = round(category.score, 2)
                # result_text = category_name + ' (' + str(probability) + ')'
                # text_location = (MARGIN + bbox.origin_x,
                #                 MARGIN + ROW_SIZE + bbox.origin_y)
                # cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                #             FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image,cropped_list,startpts_list
# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# # STEP 3: Load the input image.
# image = mp.Image.create_from_file('raw_dataset/pranav2.jpg')

# # STEP 4: Detect faces in the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the detection result. In this case, visualize it.
# image_copy = np.copy(image.numpy_view())
# annotated_image,cropped_img = visualize(image_copy, detection_result)
# rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# plt.imshow(annotated_image)
# plt.show()