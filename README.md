# DeepTAM
DeepTAM is a learnt system for keyframe-based dense camera tracking and mapping. 

If you use this code for research, please cite the following paper:

    @InProceedings{ZUB18,
        author       = "H. Zhou and B. Ummenhofer and T. Brox",
        title        = "DeepTAM: Deep Tracking and Mapping",
        booktitle    = "European Conference on Computer Vision (ECCV)",
        month        = " ",
        year         = "2018",
        url          = "http://lmb.informatik.uni-freiburg.de/Publications/2018/ZUB18"
    }
    
See the [project page](https://lmb.informatik.uni-freiburg.de/people/zhouh/deeptam/) for the paper and other material.

**Note**: Currently we only provide deployment code for the camera tracking. The mapping code will come soon.


## Setup
Current version is tested on Ubuntu 16.04 and with Python3.

```bash
# install virtualenv manager (here we use pew)
pip3 install pew

# create virtualenv
pew new deeptam

# switch to virtualenv
pew in deeptam
```

```bash
# install tensorflow 1.4.0 with gpu
pip3 install tensorflow-gpu==1.4.0

# install some python modules
pip3 install minieigen
pip3 install scikit-image
```

```bash
# clone and build lmbspecialops (use branch deeptam)
git clone -b deeptam https://github.com/lmb-freiburg/lmbspecialops.git
LMBSPECIALOPS_DIR=$PWD/lmbspecialops
cd $LMBSPECIALOPS_DIR
mkdir build
cd build
cmake ..
make

# add lmbspecialops to your PYTHON_PATH
pew add $LMBSPECIALOPS_DIR/python
```

```bash
# clone deeptam git (currently only tracking code is available)
git clone https://github.com/lmb-freiburg/deeptam.git
DEEPTAM_DIR=$PWD/deeptam

# add deeptam_tracker to your PYTHON_PATH
pew add $DEEPTAM_DIR/tracking/python
```

## Running tracking examples
```bash
# download example data
cd  $DEEPTAM_DIR/tracking/data
./download_testdata.sh

# download weights
cd $DEEPTAM_DIR/tracking/weights
./download_weights.sh
```
The basic example shows how to use DeepTAM to track the camera within one keyframe:
```bash
# run a basic example
cd $DEEPTAM_DIR/tracking/examples
python3 example_basic.py
```
The advanced example shows how to track a video sequence with multiple keyframes:
```bash
# run an advanced example
cd $DEEPTAM_DIR/tracking/examples
python3 example_advanced_sequence.py

# or run without visualization for speedup
python3 example_advanced_sequence.py --disable_vis
```

## License

deeptam is under the [GNU General Public License v3.0](LICENSE.txt)
