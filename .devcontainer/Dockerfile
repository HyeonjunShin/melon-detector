FROM mcr.microsoft.com/devcontainers/base:noble

ENV DEBIAN_FRONTEND=noninteractive
# ENV ROS_DISTRO=jazzy
# ENV USERNAME=vscode
# ENV USER_UID=1000
# ENV USER_GID=$USER_UID

# ROS2 Jazzy 설치
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-v4l2-camera 
    # && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libsoundio-dev \    
    locales \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    wget \
    git \
    libgl1 \
    libglx0 \
    libgl1-mesa-dri \
    v4l-utils \
    cython3 \
    x11-apps \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 
    # && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

COPY ../scripts/install_kinect_driver.sh /tmp/install_kinect_driver.sh
RUN chmod +x /tmp/install_kinect_driver.sh && /tmp/install_kinect_driver.sh

# RUN pip install opencv-contrib-python --break-system-packages

# # 4. 일반 사용자(ros) 생성 및 그룹 추가 (video, sudo)
# RUN if [ "$USERNAME" != "ubuntu" ]; then \
#         usermod -l $USERNAME ubuntu && \
#         usermod -d /home/$USERNAME -m $USERNAME && \
#         groupmod -n $USERNAME ubuntu; \
#     fi \
#     && usermod -aG video $USERNAME \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME


# # 6. PyTorch 설치 (Ubuntu 24.04 공식 패키지 활용)
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# # 7. Azuer Kinect 2
# RUN sudo apt-get update && sudo apt-get install -y libsoundio-dev
# # RUN pip install pyk4a --break-system-packages



# # git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
# # mkdir build && cd build
# # cmake .. \
# #     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
# #     -DCMAKE_C_FLAGS="-w -D_GNU_SOURCE" \
# #     -DCMAKE_CXX_FLAGS="-w" && \
# # /home/vscode/Azure-Kinect-Sensor-SDK/extern/azure_c_shared/src/adapters/x509_openssl.c 파일 79~84라인 수정
# #if (OPENSSL_VERSION_NUMBER >= 0x10100000L) && (OPENSSL_VERSION_NUMBER < 0x20000000L)
#                 # SSL_CTX_clear_extra_chain_certs(ssl_ctx);
# #else
#                 # // if (ssl_ctx->extra_certs != NULL)
#                 # // {
#                 # //     sk_X509_pop_free(ssl_ctx->extra_certs, X509_free);
#                 # //     ssl_ctx->extra_certs = NULL;
#                 # // }
#                 # SSL_CTX_clear_extra_chain_certs(ssl_ctx);
# #endif
# # /home/vscode/Azure-Kinect-Sensor-SDK/extern/libebml/src/src/EbmlSInteger.cpp 파일 37번 라인에 #include <limits> 추가
# # /home/vscode/Azure-Kinect-Sensor-SDK/examples/viewer/opengl/main.cpp 파일 6번라인에 #include <limits> 추가
# # /home/vscode/Azure-Kinect-Sensor-SDK/tools/k4aviewer/k4aaudiochanneldatagraph.h 10번 라인에 #include <string> 추가
# # /home/vscode/Azure-Kinect-Sensor-SDK/tools/k4aviewer/k4amicrophonelistener.cpp 11번라인에 #include <cstring> 추가
# # /home/vscode/Azure-Kinect-Sensor-SDK/tools/k4aviewer/perfcounter.h 7번 라인에 #include <string> 추가

# # make -j$(nproc) && \
# # make install

# # SDK 클론 및 빌드
# # RUN git clone https://github.com/HyeonjunShin/Azure-Kinect-Sensor-SDK.git && \
# #     mkdir -p Azure-Kinect-Sensor-SDK/build && \
# #     cd Azure-Kinect-Sensor-SDK/build && \
#     # cmake .. \
#     #   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#     #   -DCMAKE_C_FLAGS="-w -D_GNU_SOURCE" \
#     #   -DCMAKE_CXX_FLAGS="-w" && \
# #     make -j$(nproc) && \
# #     make install

# # 아까 확인했던 libdepthengine 추출 및 배치를 이어서 수행해야 함
# # RUN mkdir -p /tmp/k4a_extract && cd /tmp/k4a_extract && \
# #     wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb && \
# #     ar x libk4a1.4_1.4.1_amd64.deb && \
# #     tar -xzf data.tar.gz && \
# #     # 실행 파일이 있는 /usr/local/bin 또는 시스템 라이브러리 경로로 복사
# #     sudo cp usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0 /usr/lib/x86_64-linux-gnu/ && \
# #     cd / && rm -rf /tmp/k4a_extract

# # # 시스템 라이브러리 경로 갱신
# # RUN ldconfig

# # Privileged
# # sudo curl -sSL https://raw.githubusercontent.com/HyeonjunShin/Azure-Kinect-Sensor-SDK/refs/heads/develop/scripts/99-k4a.rules -o /etc/udev/rules.d/99-kinect.rules

# RUN apt install -y x11-apps

# # 사용자 환경 설정
# USER $USERNAME
# WORKDIR /home/$USERNAME

# # ROS2 환경 자동 로드 및 colcon 자동완성 설정
# RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc \
#     && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

# CMD ["bash"]