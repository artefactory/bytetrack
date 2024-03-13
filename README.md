# Bytetrack starter guide

This repo is a packaged version of the [ByteTrack](https://github.com/ifzhang/ByteTrack) algorithm.

<h4>
    <img width="700" alt="teaser" src="assets/traffic.gif">
</h4>

### Installation
```
pip install git+https://github.com/artefactory-fr/bytetrack.git@main
```

ByteTrack is a multi-object tracking computer vision model. Using ByteTrack, you can allocate IDs for unique objects in a video for use in tracking objects.

### Detection object with Bytetracker and YOLO
```
from bytetracker import BYTETracker
import cv2

tracker = BYTETracker(args)
for frame_id, image_filename in enumerate(frames):
    img = cv2.imread(image_filename)
    detections = model.predict(img, *yolo_args)[0]
    detections_bytetrack_format = yolo_results_to_bytetrack_format(detections)
    tracked_objects = tracker.update(detections_bytetrack_format, frame_id)
```


## Copyright

Copyright (c) 2022 Kadir Nar

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
