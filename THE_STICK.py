import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
from tracker import Tracker

"""
Installation is super simple:

pip install numpy opencv-python depthai blobconverter

done.
"""


def frameNorm(frame, bbox):
    # predection outputs are normalised and need to be rescaled to the network size
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


"""
Initialise Tracker
"""
# Variables initialization
track_colors = {}
np.random.seed(0)

# build array for all tracks and classes
track_classes = {}

tracker_KF = Tracker(dist_thresh=100,
                     max_frames_to_skip=60,
                     max_trace_length=300,
                     trackIdCount=0,
                     use_kf=True,
                     std_acc=10,
                     x_std_meas=0.25,
                     y_std_meas=0.25,
                     dt=1 / 60)

print("INITIALISED TRACKER!")

# some necessary defenitions
font = cv2.FONT_HERSHEY_SIMPLEX
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# any interaction with the OAK-D cam requires defining a pipeline, i.e., a structure of inputs, processes, and outputs
pipeline = depthai.Pipeline()

# in this test, let's grab the 4k cams rgb input, run some simple neural inference using a pre-trained mobile net
# and display the output on top of the retrieved video stream.

cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)  # resize for input of mobilenet
cam_rgb.setInterleaved(False)

# next, we'll configure the detection mobilenet
# the blob file containing configuration and weights is downloaded automatically
detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)

# Set path of the blob (NN model). (in this case, grab straight from model zoo)
# detection_nn.setBlobPath("/path/to/model.blob") # when using a locally saved model
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

# now, connect the camera's preview output to the camera
cam_rgb.preview.link(detection_nn.input)

# now transport the camera preview & network outputs from the camera to the host (this computer)
# this is done using XLinkOut nodes, which take the ouputs of the camera preview and detector as inputs respectively
# camera output
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)
# nn outout
xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# The pipeline is now completely defined! Let's initialise the connected camera with it!
# NOTE: to initialise depthai on an USB2 port use "" device = depthai.Device(pipeline, usb2Mode=True) ""

with depthai.Device(pipeline) as device:
    # access results from XLinkOut nodes
    q_rgb = device.getOutputQueue("rgb")  # the only required argument is the name of the stream
    q_nn = device.getOutputQueue("nn")

    # create placeholders for the received outputs
    frame = None
    detections = []

    # time to run some inference
    while True:
        # fill queue
        # The tryGet method returns either the latest result or None if the queue is empty.
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        # results (from both streams) will be received as 1D arrays, so require transformation upon retrieval
        # grab frame
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        # grab nn results
        # Format of MobileNetSSD [image_id, label, confidence, x_min, y_min, x_max,y_max]
        if in_nn is not None:
            detections = in_nn.detections

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
                cv2.putText(frame, labelMap[detection.label], (bbox[0], bbox[1] - 10), font, 1, (150, 20, 150))

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

cv2.destroyAllWindows()
