import os
import time
import argparse
import collections
import json

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np
import requests

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

COCO_CLASSES_LIST = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args

def dict2json(dict):
    return json.dumps(dict).replace("\\", "")

def count_cls_from_list(class_list):
    count = collections.Counter(class_list)
    # return dict2json(dict(count))
    return dict(count)

def send_data_to_endpoint(data_dict):
    headers = {'Content-Type': 'application/json'}
    endpoint = 'http://uni.soracom.io'
    payload = json.dumps(data_dict)
    print(payload)
    try: 
        response = requests.post(endpoint, data=payload, headers=headers, timeout=(3.0))
        print(response.json())
    except requests.exceptions.Timeout as e:
        print(e)
        pass

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        try:
        # if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #     break
            img = cam.read()
            if img is None:
                break
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            clss = clss.astype(np.int)
            class_no = clss.tolist()
            print(class_no)
            class_list = [COCO_CLASSES_LIST[i] for i in class_no]
            print(class_list)
            send_data_to_endpoint(count_cls_from_list(class_list))
            img = vis.draw_bboxes(img, boxes, confs, clss)
            cv2.imwrite('test.jpg', img)
            time.sleep(5)
            

            # img = vis.draw_bboxes(img, boxes, confs, clss)
            # img = show_fps(img, fps)
            # cv2.imshow(WINDOW_NAME, img)
            # toc = time.time()
            # curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            # fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            # tic = toc
            # key = cv2.waitKey(1)
            # if key == 27:  # ESC key: quit program
            #     break
            # elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            #     full_scrn = not full_scrn
            #     set_display(WINDOW_NAME, full_scrn)
        except KeyboardInterrupt:
            break

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    # open_window(
    #     WINDOW_NAME, 'Camera TensorRT YOLO Demo',
    #     cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
