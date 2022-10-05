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

In case things f*ck up, just run the following and please don't ask why.
```bash
pip install protobuf==3.20.*
```

2. conversion from a tiny yolov4 model
```bash
python convert_weights_pb.py --yolo 4 --class_names I:\THE_STICK\ant_tiny_YOLO_v4\obj.names --output_name yolov4_tiny_ants.pb --weights_file I:\THE_STICK\ant_tiny_YOLO_v4\yolov4_tiny_ants.weights --size 416 --tiny
```

3. Now, to run the tf -> OpenVINO conversion, configure the json file to match the number of classes (see yolo2openvino notes for details) and run
```bash
mo --input_model .\yolov4_tiny_ants_TF.pb --tensorflow_use_custom_operations_config .\json\yolo_v4_tiny_ants.json --batch 1 --data_type FP16 --reverse_input_channel --model_name yolov4_tiny_ants --output_dir I:\THE_STICK
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
