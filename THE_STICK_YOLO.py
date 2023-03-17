from pathlib import Path
from tracker import Tracker
import cv2
import depthai as dai
import numpy as np
import time
import blobconverter
import math


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def displayFrame(name, frame, detections):
    color = (255, 0, 0)
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                    255)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # Show the frame
    cv2.imshow(name, frame)


"""
# for stick insects
nnPath = blobconverter.from_openvino(xml="stick_tiny_YOLO_v4/yolov4_tiny_sticks.xml",
                                     bin="stick_tiny_YOLO_v4/yolov4_tiny_sticks.bin",
                                     data_type="FP16",
                                     shaves=6,
                                     version="2021.3",
                                     use_cache=True)
"""
# for ants
nnPath = blobconverter.from_openvino(xml="ant_tiny_YOLO_v4/yolov4_tiny_ants_416.xml",
                                     bin="ant_tiny_YOLO_v4/yolov4_tiny_ants_416.bin",
                                     data_type="FP16",
                                     shaves=6,
                                     version="2021.3",
                                     use_cache=True)


if not Path(nnPath).exists():
    import sys

    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

"""
Initialise Tracker
"""
# Variables initialization
track_colors = {}
np.random.seed(0)

# build array for all tracks and classes
track_classes = {}

tracker_KF = Tracker(dist_thresh=250,
                     max_frames_to_skip=60,
                     max_trace_length=300,
                     trackIdCount=0,
                     use_kf=True,
                     std_acc=10,
                     x_std_meas=0.5,
                     y_std_meas=0.5,
                     dt=1 / 60)

max_allowed_deviation = 40

print("INITIALISED TRACKER!")

# some necessary defenitions
font = cv2.FONT_HERSHEY_SIMPLEX

labelMap = ["insect"]

syncNN = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
# camRgb.setPreviewSize(320, 320)
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(60)

# Network specific settings
detectionNetwork.setConfidenceThreshold(0.7)
detectionNetwork.setNumClasses(1)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])

# for 320 x 320
# detectionNetwork.setAnchorMasks({"side20": [1, 2, 3], "side10": [3, 4, 5]})
# 416 x 416
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        centres = []
        bounding_boxes = []
        predicted_classes = []

        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame=frame,
                                 bbox=(detection.xmin,
                                       detection.ymin,
                                       detection.xmax,
                                       detection.ymax))
                centres.append([[(bbox[0] + bbox[2]) / 2],
                                [(bbox[1] + bbox[3]) / 2]])
                bounding_boxes.append([bbox[0],
                                       bbox[2],
                                       bbox[1],
                                       bbox[3]])
                predicted_classes.append(labelMap[detection.label])

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (150, 20, 150), 2)
                cv2.putText(frame, labelMap[detection.label], (bbox[0], bbox[1] - 10), font, 0.4, (150, 20, 150))

            if len(centres) > -1:

                # Track object using Kalman Filter
                tracker_KF.Update(centres,
                                  predicted_classes=predicted_classes,
                                  bounding_boxes=bounding_boxes)

                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker_KF.tracks)):
                    if len(tracker_KF.tracks[i].trace) > 1:
                        mname = "track_" + str(tracker_KF.tracks[i].track_id)

                        if mname not in track_colors:
                            track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()

                        # draw direction of movement onto footage
                        x_t, y_t = tracker_KF.tracks[i].trace[-1]
                        tracker_KF_velocity = 5 * (tracker_KF.tracks[i].trace[-1] - tracker_KF.tracks[i].trace[-2])
                        x_t_future, y_t_future = tracker_KF.tracks[i].trace[-1] + tracker_KF_velocity * 0.1
                        cv2.arrowedLine(frame, (int(x_t), int(y_t)), (int(x_t_future), int(y_t_future)),
                                        (np.array(track_colors[mname]) - np.array([70, 70, 70])).tolist(), 3,
                                        tipLength=0.75)

                        for j in range(len(tracker_KF.tracks[i].trace) - 1):

                            # Draw trace line on preview
                            x1 = tracker_KF.tracks[i].trace[j][0][0]
                            y1 = tracker_KF.tracks[i].trace[j][1][0]
                            x2 = tracker_KF.tracks[i].trace[j + 1][0][0]
                            y2 = tracker_KF.tracks[i].trace[j + 1][1][0]
                            if mname not in track_colors:
                                track_colors[mname] = np.random.randint(low=100, high=255, size=3).tolist()
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[mname], 2)

                        cv2.putText(frame,
                                    mname,
                                    (int(x1) - int(30 / 2),
                                     int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    track_colors[mname], 2)


            cv2.imshow("preview", frame)

            if cv2.waitKey(1) == ord('q'):
                break