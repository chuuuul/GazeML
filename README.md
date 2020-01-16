# 시선추적을 활용한 방범시스템 

### 문제 제기
 아직도 우리 사회에서는 근로자를 위협하는 범죄들이 발생하고 있다. 어떠한 장치들이 근로자들을 보호하고 있는지 조사하였고, 현재 설치되어 있는 방범 시스템들이 근로자들을 보호하는데에 부족하다고 느꼈다. 특히 1인 근로자들은 위험에 많이 노출 되어 있었다. 편의점의 경우 비상전화가 있었지만, 위협을 받는 상황에 전화를 할 수 없는 등 실효성의 문제와 오작동의 문제가 있었다. 또한 택시의 경우 차 위에 불이 빨갛게 점등되는 장치가 전부였다. 이 점등장치를 본 사람들이 신고를 대신 해주어야 하는 불편함이 있었으며, '비상 점멸장치'의 의미를 시민들이 알고 있어야 한다는 문제가 있다.

### 아이디어
  우리는 좀 더 은밀하고 빠르고 정확한 방법의 신고 시스템을 구축하기 위해 이 프로젝트를 진행 하였다. 우리의 아이디어는 시선을 이용하여 신고를 하는 시스템이다. 시선은 순식간에 많은 정보를 전달 할 수 있으며 신체적인 움직임이 최소화 되어 은밀하게 신고를 할 수 있기 때문이다.


### 컨셉
![컨셉]( https://github.com/chuuuul/arto_eye/blob/master/GitSource/designIdea.png)


### 흐름도
![흐름도]( https://github.com/chuuuul/arto_eye/blob/master/GitSource/flow.png )


### 실행 화면

![방범시스템앱](https://github.com/chuuuul/arto_eye/blob/master/GitSource/security_app.png)
방범 시스템 어플리케이션

### 실행 방법
OS : Ubuntu
H/W : USB Webcam
1. elg_demo.py를 실행시켜 캘리브레이션을 진행시킨다.
2. security_app.py를 실행시켜 시선추적으로 패턴을 입력한다.



</br></br>


# 장애인을 위한 시선추적 기반 의사소통 시스템

### 목적
 이 프로젝트는 의사소통과 행동에 제약이 있는 신경 마비 장애인을 위한 어플리케이션이다. 눈동자의 움직임만으로 일상생활에서 사용하는 의사소통을 할 수 있게 도와준다. 이 프로젝트는 전문가용 카메라가 아닌 일반적으로 접할 수 있는 웹캠만 필요하기 때문에 경제적 부담이 적게 의사소통 시스템을 갖출 수 있다는 장점이 있다.



### 실행 화면
 
![의사소통앱](https://github.com/chuuuul/arto_eye/blob/master/GitSource/community_app.png)
의소소통 어플리케이션


### 실행 방법
OS : Ubuntu
H/W : USB Webcam
1. elg_demo.py를 실행시켜 캘리브레이션을 진행시킨다.
2. community_app.py를 실행시켜 앱을 실행시킨다.
3. 의사소통의 대분류를 시선으로 응시하여 선택하면 의사소통의 문구를 선택 할 수 있으며, 선택한 문구를 읽어주게 된다.



############################################################################################

# GazeML
A deep learning framework based on Tensorflow for the training of high performance gaze estimation.

*Please note that though this framework may work on various platforms, it has only been tested on an Ubuntu 16.04 system.*

*All implementations are re-implementations of published algorithms and thus provided models should not be considered as reference.*

This framework currently integrates the following models:

## ELG

Eye region Landmarks based Gaze Estimation.

> Seonwook Park, Xucong Zhang, Andreas Bulling, and Otmar Hilliges. "Learning to find eye region landmarks for remote gaze estimation in unconstrained settings." In Proceedings of the 2018 ACM Symposium on Eye Tracking Research & Applications, p. 21. ACM, 2018.

- Project page: https://ait.ethz.ch/projects/2018/landmarks-gaze/
- Video: https://youtu.be/cLUHKYfZN5s

## DPG

Deep Pictorial Gaze Estimation

> Seonwook Park, Adrian Spurr, and Otmar Hilliges. "Deep Pictorial Gaze Estimation". In European Conference on Computer Vision. 2018

- Project page: https://ait.ethz.ch/projects/2018/pictorial-gaze

*To download the MPIIGaze training data, please run `bash get_mpiigaze_hdf.bash`*

*Note: This reimplementation differs from the original proposed implementation and reaches 4.63 degrees in the within-MPIIGaze setting. The changes were made to attain comparable performance and results in a leaner model.*

## Installing dependencies

Run (with `sudo` appended if necessary),
```
python3 setup.py install
```

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case, the sequence of commands would be similar to:
```
    mkvirtualenv -p $(which python3) myenv
    python3 setup.py install
```

when using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

### Tensorflow
Tensorflow is assumed to be installed separately, to allow for usage of [custom wheel files](https://github.com/mind/wheels) if necessary.

Please follow the official installation guide for Tensorflow [here](https://www.tensorflow.org/install/).

## Getting pre-trained weights
To acquire the pre-trained weights provided with this repository, please run:
```
    bash get_trained_weights.bash
```

## Running the demo
To run the webcam demo, perform the following:
```
    cd src
    python3 elg_demo.py
```

To see available options, please run `python3 elg_demo.py --help` instead.

## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `outputs/` - any output for a model will be placed here, including logs, summaries, and checkpoints.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - neural network definitions
    * `util/` - utility methods
