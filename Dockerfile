# base docker image can be searched in https://hub.docker.com/search?q=
# FROM ubuntu:20.04
# FROM osrf/ros-noetic-desktop
# FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG USR_NAME=shunsuke-m

# avoid stopping in apt-get
ENV DEBIAN_FRONTEND=noninteractive

#####################################################
# General settings
#####################################################
RUN apt-get -y update && apt-get install -y \
    tzdata \
    gosu \
    gawk \
    gcc \
    make \
    cmake \
    sudo \
    ffmpeg \
    unzip \
    net-tools \
    iputils-ping \
    dnsutils \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    gedit \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git \
    python3-openssl \
    terminator \
    locales \
    xterm \
    dbus-x11 \
    libx11-dev



RUN mkdir -p /home/${USR_NAME}/workspace



#####################################################
# Locale
#####################################################
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ=Asia/Tokyo 



# #####################################################
# # ROS settings
# #####################################################
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN apt-get update && apt-get install -y libfcl* \
#                        libglew-dev \
#                        ros-noetic-desktop-full \
#                        ros-noetic-joy \
#                        ros-noetic-gazebo* \
#                        ros-noetic-moveit* \
#                        ros-noetic-usb-cam* \
#                        ros-noetic-image-view* \
#                        ros-noetic-cv-camera* \
#                        ros-noetic-joint* \
#                        ros-noetic-graph* \
#                        ros-noetic-ros-controller* \
#                        ros-noetic-joy-teleop* \
#                        ros-noetic-eigen* \
#                        ros-noetic-socketcan-bridge \
#                        ros-noetic-rosbridge-server* \
#                        ros-noetic-octomap* \
#                        ros-noetic-geometric* \
#                        ros-noetic-object-recognition* \
#                        ros-noetic-srdfdom* \
#                        ros-noetic-ompl* \
#                        ros-noetic-sbpl* \
#                        ros-noetic-map-server* \
#                        ros-noetic-warehouse-ros* \
#                        ros-noetic-spacenav* \
#                        ros-noetic-soem* \
#                        ros-noetic-geodesy \
#                        python3-pip python3-vcstool python3-pyqt5 \
#                        pyqt5-dev-tools \
#                        libbluetooth-dev libspnav-dev \
#                        libcwiid-dev \
#                        python3-rosdep \
#                        python3-serial \
#                        cmake gcc g++ qt5-qmake qtbase5-dev \
#                        libusb-dev libftdi-dev \
#                        libsdl-dev \
#                        libsdl-image1.2-dev \
#                        python3-defusedxml python3-vcstool \
#                        ros-noetic-control-toolbox \
#                        ros-noetic-pluginlib \
#                        ros-noetic-trajectory-msgs \
#                        ros-noetic-control-msgs \
#                        ros-noetic-std-srvs \
#                        ros-noetic-nodelet \
#                        ros-noetic-urdf \
#                        ros-noetic-rviz \
#                        ros-noetic-kdl-conversions \
#                        ros-noetic-tf2-sensor-msgs \
#                        ros-noetic-pcl-ros \
#                        ros-noetic-navigation \
#                        ros-noetic-sophus \
#                        ros-noetic-rosserial \
#                        ros-noetic-rosserial-arduino \
#                        ros-noetic-rosserial-python
						


# RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN echo "" >> /root/.bashrc
# RUN echo "# ROS settings" >> /root/.bashrc
# RUN echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> /root/.bashrc
# RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
# RUN echo "source /home/${USR_NAME}/catkin_ws/devel/setup.bash" >> /root/.bashrc

# RUN rosdep init && rosdep update



# #####################################################
# # realsense settings
# #####################################################
# RUN apt-get update 
# RUN apt-get install -y software-properties-common
# RUN apt-key adv --keyserver keyserver.ubuntu.com \
#                 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
#     apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 \
#                 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
# RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
# RUN apt-get install -y librealsense2-dkms \
#                        librealsense2-utils \
#                        librealsense2-dev \
#                        librealsense2-dbg
# RUN apt-get install -y ros-noetic-realsense2-camera



#####################################################
# pyenv settings
#####################################################
ENV PYENV_HOME /home/${USR_NAME}
ENV PYENV_ROOT $PYENV_HOME/.pyenv
ENV PATH $PYENV_HOME/.pyenv/shims:$PATH
ENV PATH $PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_HOME/.pyenv

RUN echo "" >> /root/.bashrc
RUN echo "# pyenv settings" >> /root/.bashrc
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_HOME/.pyenv/shims:$PATH"' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

RUN pyenv install 3.11.11
RUN pyenv rehash
RUN pyenv global 3.11.11
 
RUN pip install --upgrade pip

RUN pip install matplotlib \
                ipython \
                scikit-learn \
                scipy \
                opencv-python==4.7.0.72 \
                opencv-contrib-python==4.7.0.72 \
                sympy \
                seaborn\
                pandas \
                h5py \
                ipdb 

# for ROS
# RUN pip install catkin_pkg empy pyaml rospkg numpy-quaternion defusedxml

RUN pip install --upgrade pip setuptools wheel uv

RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git /home/${USR_NAME}/workspace/openpi

RUN cd /home/${USR_NAME}/workspace/openpi && uv sync

#####################################################
# ipython settings
#####################################################

RUN git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git /home/${USR_NAME}/.dynamixel
RUN cd /home/${USR_NAME}/.dynamixel/python && python setup.py install


#####################################################
# ipython settings
#####################################################
RUN mkdir -p /home/${USR_NAME}/.ipython/profile_default
RUN touch /home/${USR_NAME}/.ipython/profile_default/00-startup.py
RUN echo "import os, sys" >> /home/${USR_NAME}/.ipython/profile_default/00-startup.py
RUN echo "import matplotlib.pyplot as plt" >> /home/${USR_NAME}/.ipython/profile_default/00-startup.py
RUN echo "import numpy as np" >> /home/${USR_NAME}/.ipython/profile_default/00-startup.py



#####################################################
# vim settings
#####################################################
RUN apt-get update && apt-get install -y vim-gtk3
RUN mkdir /home/${USR_NAME}/.vim
RUN git clone https://github.com/tomasr/molokai /home/${USR_NAME}/.vim
COPY assets/vimrc.txt /home/${USR_NAME}/.vimrc


RUN pip install --upgrade pip setuptools wheel
RUN pip uninstall -y gym gym-robotics mujoco-py || true
# RUN apt-get install -y ros-noetic-rosserial*

# #####################################################
# # PiPER ROS Setting
# #####################################################
# RUN pip3 install python-can
# RUN pip3 install piper_sdk
# RUN pip3 install empy==3.3.4
# RUN apt -y install ethtool
# RUN apt -y install iproute2
# RUN apt -y install can-utils


# RUN useradd -ms /bin/bash $USR_NAME
# # 作業ディレクトリ指定
# WORKDIR /home/$USR_NAME
# # ##公式のROSパッケージ
# # RUN git clone https://github.com/agilexrobotics/piper_ros.git
# # RUN apt -y install python3-wstool python3-catkin-tools python3-rosdep ros-noetic-ruckig
# # RUN apt-get -y install ros-noetic-eigen-stl-containers ros-noetic-geometric-shapes ros-noetic-moveit-msgs ros-noetic-srdfdom ros-noetic-pybind11-catkin
# # RUN apt-get -y install ros-noetic-moveit-resources-panda-moveit-config ros-noetic-ompl ros-noetic-warehouse-ros ros-noetic-eigenpy ros-noetic-rosparam-shortcuts
# # ## ビルド
# # RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
# #     cd piper_ros && \
# #     catkin_make"

# # RUN echo "alias set_init='cd ~/docker-PiPER/double_PiPER && \
# #                           sudo sh package_install.sh && \
# #                           catkin_make && \
# #                           source devel/setup.bash'" >> /root/.bashrc

# RUN echo "alias set_sdk='cd ~ && \
#                             git clone -b 1_0_0_beta https://github.com/agilexrobotics/piper_sdk.git && \
#                             cd ~/piper_sdk && \
#                             pip3 install . && \
#                             cd ~/docker_PiPER_env_ver2/double_PiPER && \
#                             catkin_make && \
#                             source devel/setup.bash'" >> /root/.bashrc

# RUN echo "alias sorc_devel='cd ~/docker_PiPER_env_ver2/double_PiPER && \
#                           source devel/setup.bash'" >> /root/.bashrc


# RUN echo "alias init_ethernet='sudo ethtool -i can0 | grep bus && \
#                                 sudo ethtool -i can1 | grep bus && \
#                                 sudo ip link set can0 down && \
#                                 sudo ip link set can1 down && \
#                                 ip link show can0 && \
#                                 ip link show can1" >> /root/.bashrc
# RUN pip install "gymnasium-robotics==1.4.2"

RUN pip uninstall -y numpy
RUN pip install --no-cache-dir "numpy==1.26.4"

RUN pip uninstall -y opencv-python opencv-contrib-python
RUN pip install --no-cache-dir "opencv-python==4.8.1.78"
RUN pip install pyyaml

# pyenv 側の python に torch を入れる
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# openpi 側の uv 環境にも torch を入れる
RUN cd /home/${USR_NAME}/workspace/openpi && \
    uv pip install --python .venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#####################################################
# Entry script - This will also run terminator
#####################################################
### terminator setting
COPY assets/config /


### user group settings
COPY assets/entrypoint_setup.sh /
# COPY assets/entrypoint_setup.sh /entrypoint_setup.sh
ENTRYPOINT ["/entrypoint_setup.sh"] 

# SHELL ["/bin/bash", "-l", "-c"]
CMD ["terminator"]



