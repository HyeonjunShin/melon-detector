#!/bin/bash

set -e

cd /tmp
REPO_DIR="Azure-Kinect-Sensor-SDK"
# 1. 저장소 클론 및 이동
if [ ! -d "$REPO_DIR" ]; then
    git clone -b v1.4.2 https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
fi
cd $REPO_DIR

# 2. 서브모듈 다운로드를 위한 1차 CMake 실행
mkdir -p build && cd build
echo "--- 종속성 서브모듈 다운로드 중 (1차 CMake) ---"
cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS="-w -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-w"

# 3. 소스 코드 수정 (Patching)
cd ..
echo "--- 소스 코드 수정 시작 ---"

sed -i '79,83d' "extern/azure_c_shared/src/adapters/x509_openssl.c"
sed -i '79i \                SSL_CTX_clear_extra_chain_certs(ssl_ctx);' "extern/azure_c_shared/src/adapters/x509_openssl.c"

sed -i 's/OPENSSL_VERSION_NUMBER < 0x20000000L/OPENSSL_VERSION_NUMBER < 0x30000000L/g' "extern/azure_c_shared/src/adapters/x509_openssl.c"

# [B] 헤더 파일 추가 (중복 체크 포함 함수)
patch_header() {
    FILE=$1; HEADER=$2; LINE=$3
    if [ -f "$FILE" ]; then
        if ! grep -q "$HEADER" "$FILE"; then
            sed -i "${LINE}i ${HEADER}" "$FILE"
            echo "[$FILE] ${HEADER} 추가 완료 (Line ${LINE})."
        else
            echo "[$FILE] ${HEADER}가 이미 존재합니다."
        fi
    fi
}

# 요청하신 모든 헤더 추가 작업
patch_header "tools/k4amicrophonelistener.cpp" "#include <cstring>" 7
patch_header "extern/libebml/src/src/EbmlSInteger.cpp" "#include <limits>" 37
patch_header "examples/viewer/opengl/main.cpp" "#include <limits>" 6
patch_header "tools/k4aviewer/k4aaudiochanneldatagraph.h" "#include <string>" 10
patch_header "tools/k4aviewer/perfcounter.h" "#include <string>" 7
patch_header "tools/k4aviewer/k4amicrophonelistener.cpp" "#include <cstring>" 11

# (기존에 요청하셨던 11번 라인 cstring은 7번 라인과 중복될 수 있어 7번 우선 적용)

echo "--- 모든 소스 패치 완료 ---"
sudo cp scripts/99-k4a.rules /etc/udev/rules.d/

# 4. 최종 빌드 및 설치
cd build
echo "--- 컴파일 시작 (make) ---"
make -j$(nproc)

echo "--- 시스템 설치 (sudo make install) ---"
sudo make install

# 5. Depth Engine (libdepthengine.so) 추출 및 설치
echo "--- Depth Engine 설치 시작 ---"
TEMP_DIR="/tmp/k4a_extract"
mkdir -p $TEMP_DIR && cd $TEMP_DIR

# .deb 패키지 다운로드 및 압축 해제
wget -q https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
ar x libk4a1.4_1.4.1_amd64.deb
tar -xzf data.tar.gz

# 라이브러리 복사 및 심볼릭 링크 생성
sudo cp usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0 /usr/lib/x86_64-linux-gnu/
sudo ln -sf /usr/lib/x86_64-linux-gnu/libdepthengine.so.2.0 /usr/lib/x86_64-linux-gnu/libdepthengine.so.2
sudo ln -sf /usr/lib/x86_64-linux-gnu/libdepthengine.so.2 /usr/lib/x86_64-linux-gnu/libdepthengine.so

sudo apt install -y python3-pip
sudo apt install -y libsoundio-dev
pip3 install pyk4a --break-system-packages

# 정리 및 라이브러리 갱신
cd / && rm -rf $TEMP_DIR && rm -rf $REPO_DIR
sudo ldconfig

echo "--- 모든 작업이 성공적으로 완료되었습니다! ---"
