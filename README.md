# Bytetrack starter guide
***** Overview *****

This repo is a packaged version of the [ByteTrack](https://github.com/ifzhang/ByteTrack) algorithm.

<h4>
    <img width="700" alt="teaser" src="assets/traffic.gif">
</h4>

### Installation
```
pip install bytetracker
```

This guide offers a beginner-friendly introduction to utilizing ByteTrack for car detection within video footage. ByteTrack is an advanced algorithm that extends the capabilities of the YOLO (You Only Look Once) model for object detection, with a particular emphasis on efficiently and accurately tracking multiple objects, such as cars, across video frames.

**Leveraging YOLOv5 and ByteTrack**

In this implementation, we are leveraging the powerful object detection capabilities of YOLOv5, available through the YOLOv5-Pip package, alongside the innovative tracking techniques provided by ByteTrack. These tools are instrumental in our approach to identifying and tracking cars, showcasing how they can be utilized in practical applications.

## Car Detection with ByteTrack - An Introductory Guide

In our notebook, we demonstrate how to set up and apply ByteTrack for car detection in video footage. The process involves:

1. **Preprocessing the video:** Extracting frames from the video to prepare for detection.
2. **Applying YOLOv5:** Detecting cars in each frame using YOLOv5.
3. **Tracking with ByteTrack:** Using ByteTrack to maintain consistent tracking of each car across the video frames.

This practical example illustrates the effectiveness of combining YOLOv5's detection capabilities with ByteTrack's with a basic example of car detection and tracking.

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
