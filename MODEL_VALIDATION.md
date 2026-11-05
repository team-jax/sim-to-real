# URDF 외형 및 초기 겹침 조사

## 변경 내용

이전 viewer는 충돌 STL만 표시했다. 원본 DAE를 OBJ로 메모리에서 변환해
외형을 표시하도록 수정했다. scene 변환을 적용하고 DAE의 meter=1, Z_UP을 확인한다.
다른 좌표 규약은 자동 추정하지 않고 오류로 처리한다.
원본 파일과 collision scale=0.01은 수정하지 않았다.
DAE 내부 부품별 색상·재질은 아직 보존하지 않으며 URDF material 색상을 사용한다.

외형은 충돌하지 않는 group 1, 충돌 형상은 group 0으로 구분한다.
현재 장면은 URDF 베이스 기준이며 Isaac의 월드 베이스 배치와 책상은 추가하지 않았다.

## 수치 확인

- URDF의 joint origin(rpy/xyz)·axis로 별도 FK를 계산했다.
- 0°, 사전 파지, ㄴ자 자세에서 16개 링크의 위치·회전 행렬이 MuJoCo와 절대오차 1e-8 이내에서 일치했다.
- 외형 추가 전후 body mass와 접촉 결과가 같았다.
- 외형 geom 16개는 contype=0, conaffinity=0으로 접촉 계산에 참여하지 않는다.
- 이 검사는 원본 URDF와 변환 사이의 일치 검사다. 실물이나 Isaac의 USD 결과와의 비교는 아니다.

## 겹침 원인에 관한 근거

초기 자세에서 RotationBody–Finger_1_1 약 12.91 mm, Shoulder_Pitch–Shoulder_Roll 약 1.47 mm의 겹침이 검출된다.
사전 파지와 ㄴ자 자세에서는 손의 해당 겹침이 사라지고 어깨 겹침은 남는다.

RotationBody와 Finger_1_1 충돌 STL은 각각 12개 삼각형의 단순 형상이다.
스케일 적용 후 바운딩 박스가 외형 DAE의 바운딩 박스와 거의 일치한다.
이는 STL이 세부 외형을 생략한 근사 형상이라는 근거이며, 실제 표면 간섭의 증거는 아니다.
MuJoCo의 메시 충돌 근사와 접합부 형상도 결과에 영향을 줄 수 있다.
겹침의 실제 기계적 의미는 외형 중첩과 기구 확인 전 확정하지 않는다.

원본 JaxIsaacLab의 아래 파일도 읽어 비교했다.

`source/isaaclab_assets/isaaclab_assets/robots/jax_v2_assembly_1006.py`

여기에는 `self_collision=False`, `enabled_self_collisions=False`가 설정되어 있다.
따라서 현재 MuJoCo 자기 충돌 검사는 원본 Isaac보다 추가된 검사이며,
이 실패를 원본 정책이 실패했다는 뜻으로 해석하면 안 된다.
런타임에서 설정을 덮어쓰는지와 실제 학습 환경 설정까지는 아직 확인하지 않았다.

## 남은 확인

- 화면에서 원본 외형과 CANWeb의 모습 비교
- 접합부와 손 겹침 부위의 외형 표면 확인
- 원본 학습·렌더 환경의 최종 충돌 설정 대조
- 이후 정책 재현 모드와 자기 충돌 검사 모드를 명시적으로 분리

자기 충돌 테스트는 현재도 실패 상태로 유지한다. 형상 변경이나 충돌 제외는 적용하지 않았다.
이 실행 환경에서는 CoreGraphics 연결 오류로 offscreen 이미지를 렌더링하지 못했다.
수치 검사는 완료했으며 시각 검사는 `inspect_model` viewer에서 확인해야 한다.
