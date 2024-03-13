# Bytetrack starter guide
***** Overview *****

This repo is a packaged version of the [ByteTrack](https://github.com/ifzhang/ByteTrack) algorithm.

<h4>
    <img width="700" alt="teaser" src="assets/traffic.gif">
</h4>

### Installation
```
pip install git+https://github.com/artefactory-fr/bytetrack.git@main
```

ByteTrack is a multi-object tracking computer vision model. 
Using ByteTrack, you can allocate IDs for unique objects in a video for use in tracking objects.

## Copyright

Copyright (c) 2022 Kadir Nar

## Reference and Acknowledgment:

- **YOLOv5-Pip:** We use YOLOv5 for its state-of-the-art object detection capabilities, making it possible to detect cars in each frame of the video. The YOLOv5-Pip package simplifies the integration of YOLOv5 into Python projects. For more details, visit the [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).

- **ByteTrack:** ByteTrack excels in tracking objects over time using the detections provided by YOLOv5. It is particularly effective in maintaining identities across frames, even in challenging conditions where objects may be occluded or move unpredictably. For more information, check out the [ByteTrack GitHub repository](https://github.com/ifzhang/ByteTrack).

## ByteTrack License

ByteTrack is licensed under the MIT License. See the [LICENSE](LICENSE) file and the [ByteTrack repository](https://github.com/bytedance/ByteTrack) for more information.


### Citation
```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
