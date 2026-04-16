import zenoh
import numpy as np
import time

def handler(sample):
    """
    수신된 ZBytes 페이로드를 문자열로 변환하여 파싱합니다.
    """
    try:
        # Zenoh 1.0+ 방식: payload를 문자열로 변환
        # .to_decode() 혹은 str(sample.payload) 사용 가능
        payload_str = sample.payload.to_string() 
        data = payload_str.split()
        
        if len(data) < 18:
            return

        rank = int(data[0])
        score = float(data[1])
        
        # 4x4 TF 행렬 복원
        tf = np.array(data[2:], dtype=float).reshape((4, 4))
        
        print(f"\n--- 🍈 Received Target Rank #{rank} ---")
        print(f"Priority Score: {score:.4f}")
        print(f"Transformation Matrix (TF):\n{tf}")
        
    except Exception as e:
        print(f"❌ 데이터 파싱 에러: {e}")

def run_subscriber():
    conf = zenoh.Config()
    
    # 세션 연결
    print("🌿 Zenoh 수신 세션 연결 중...")
    z_session = zenoh.open(conf)

    # 송신측과 일치하는 토픽 (앞에 / 없음)
    topic = "detector/response"
    sub = z_session.declare_subscriber(topic, handler)

    print(f"✅ 구독 시작: [{topic}] (Ctrl+C로 종료)")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 수신 종료")
    finally:
        z_session.close()

if __name__ == "__main__":
    run_subscriber()