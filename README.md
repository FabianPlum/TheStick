# TheStick
Real-time-hand-held-ant-detection-and-tracking-using-opencv-cameras-in-the-field-the-movie-TM

## OAK-D Environment Setup and example Execution

depthai environment for all actual inference things and the initial darknet -> tf conversion
```bash
conda create --name depthai python=3.7
conda activate depthai
```

then install additional dependencies
```bash
pip install depthai
pip install numpy==1.16.6
pip install tensorflow==1.14.0
pip install blobconverter==1.2.7
pip install openvino-dev[tensorflow]==2021.3
pip install Pillow
```

## Using our pre-trained models to run Oak-D demo tracking

To use our pre-trained demo models for ants and stick insects, you just need to run the following commands:

```bash
conda activate depthai

# to use a different model, open the file and change nnPath
python THE_STICK_YOLO.py
```

If you need to change the network input resolution (416 x 416 by default), you will need to repeat the conversion
process outlined below.

___
## Converting Models from darknet to OpenVINO Oak-D

1. In case things don't work as expected, just run the following and please don't ask why.
```bash
pip install protobuf==3.20.*
```

2. conversion from a tiny yolov4 model
```bash
python yolo2openvino\convert_weights_pb.py --yolo 4 --class_names ant_tiny_YOLO_v4\obj.names --output .\ant_tiny_YOLO_v4\yolo4tiny416.pb --tiny -h 416 -w 416 -a 10,14,23,27,37,58,81,82,135,169,344,319 --weights_file .\ant_tiny_YOLO_v4\yolov4_tiny_ants.weights
```

3. Now, to run the tf -> OpenVINO conversion, configure the json file to match the number of classes (see yolo2openvino notes for details).
run the following command from ```...\anaconda3\envs\depthai\Lib\site-packages\mo.py```
```bash
python python C:\Users\Legos\anaconda3\envs\depthai\Lib\site-packages\mo.py --input_model .\ant_tiny_YOLO_v4\yolo4tiny416.pb --tensorflow_use_custom_operations_config .\yolo2openvino\json\yolo_v4_tiny_ants.json --batch 1 --data_type FP16 --reverse_input_channel --model_name yolov4_tiny_ants_416 --output_dir .\ant_tiny_YOLO_v4
```

4. Add the following lines to final depthai pipeline setup to convert the model at run time (setting use_cashe=True  means the next time the model is run, the already compiled version is executed, so we don't need to re-compile upon every execution)
```bash
blob_path = blobconverter.from_openvino(
    xml=path_to_xml,
    bin=path_to_bin,
    data_type="FP16",
    shaves=6,
    version="2021.3",
    use_cache=True
)
```

alternatively, you can pre-compile the blob file online
https://blobconverter.luxonis.com/

Select OpenVINO 2021.3 > OpenVINO Model > Continue, upload .xml and .bin, and convert.

## Download trained networks and config files here:

**[YOLO](https://github.com/AlexeyAB/darknet) Networks**

* [tiny-yolov4 - single class ant detector (trained on synthetic data)](https://drive.google.com/drive/folders/1MS-gLpiWfPuGOIwQKvsqZsDFK29vZe9C?usp=share_link)
* [tiny-yolov4 - single class stick insect detector (trained on synthetic data)](https://drive.google.com/drive/folders/1WxRES6dMZblyQkBPMn8ZSmjOM_FNAdUY?usp=share_link)
