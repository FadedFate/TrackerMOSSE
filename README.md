# TrackerMOSSE ![](https://img.shields.io/badge/license-MIT-blue)
A MOSSE implementation in python, adapted from the matlab version [mosse-tracker-master](https://github.com/amoudgl/mosse-tracker)

## Reference
Details regarding the tracking algorithm can be found in the following paper:

[Visual Object Tracking using Adaptive Correlation Filters](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5539960).   
David S. Bolme, J. Ross Beveridge, Bruce A. Draper, Yui Man Lui.   
Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010.

## Usage
The implementation uses got10k for tracking performance evaluation. which ([GOT-10k toolkit](https://github.com/got-10k/toolkit)) is a visual tracking toolkit for VOT evaluation on main tracking datasets.
* Run test.py to evaluate on OTB or VOT dataset.
```cmd
>> python test.py
```
