# sim-to-real 검증

이 저장소를 검증 코드의 Git 루트로 사용합니다.
원격 저장소: https://github.com/team-jax/sim-to-real

배포 구조와 구현 순서: [DEPLOYMENT_SPEC.md](DEPLOYMENT_SPEC.md)

제어 흐름 그림과 큐브 배치 원점: [ARCHITECTURE.md](ARCHITECTURE.md)

원본 코드 대조 결과와 엄지 관측 수정: [ISAAC_COMPARISON.md](ISAAC_COMPARISON.md)

## 실제 신경망으로 자동 접근·파지 시도

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/mjpython -m deploy.run --layout edge10
```

창을 클릭하고 **SPACE**를 누르면 두 ONNX 정책을 사용한 전체 체인이 시작됩니다.
SPACE를 다시 누르면 일시 정지합니다. 수동 장면의 0/1/2 자세 선택은 사용하지 않습니다.
이 실행기는 시뮬 전용이며 CAN/실물 연결 코드는 없습니다.

큐브 중심의 수평 위치를 직접 지정할 때는 **월드 좌표, m 단위**입니다.
큐브는 사용자 지정 **한 변 4 cm, 50 g**입니다.
원본 마찰은 큐브·손 정지 0.4 / 동 0.35, 책상 정지 1.2 / 동 1.0입니다.
MuJoCo에서는 원본의 동마찰과 평균 결합을 명시적 접촉쌍으로 근사합니다.
큐브–손은 0.35, 큐브–책상 및 로봇–책상은 0.675이며 정지마찰의 별도 전환은 재현하지 않습니다.
큐브 중심 높이는 책상 위 2 cm로 자동 결정합니다.
학습 때의 5.5 cm 큐브와 크기가 다르므로 기존 조회표·정책의 파지 성공은 다시 확인해야 합니다.

```sh
# edge10 배치 기준점 (-0.285, 0.260)에 놓기
.venv/bin/mjpython -m deploy.run --layout edge10 --cube-x -0.285 --cube-y 0.260
# 같은 책상에서 안쪽으로 3 cm, 옆으로 2 cm 이동
.venv/bin/mjpython -m deploy.run --layout edge10 --cube-x -0.315 --cube-y 0.280
# 로봇 중심부터 책상까지 10 cm인 배치 (실험 범위)
.venv/bin/mjpython -m deploy.run --layout center10
# 화면 없이 실행하고 결과 기록
.venv/bin/python -m deploy.run --headless --layout edge10
```

카메라 좌표를 그대로 넣으면 안 됩니다. 로봇 배치에 맞는 월드 좌표로 바꾼 큐브 중심을 넣어야 합니다.
현재 입력은 시작 시점의 위치이며 실행 중 물체 상태는 MuJoCo에서 읽습니다.
팀원의 카메라 연결, 임의 크기·방향의 직육면체 파지는 아직 지원하지 않습니다.

연결된 구성:

- 접근: Catmull-Rom 경로 + 위치·높이 조회표 + `raise_v25_1197.onnx`
- 정지 확인 → 2.5 cm 밀어 넣기 → `grasp_lift_13075.onnx`
- 큐브 접촉력으로 RA12P 가상 ADC 생성 (엄지는 정책 입력에서 힘 크기, 닫기에서는 수직 성분)
- 조임 힘·끝마디 제어 + 손끝 중점 DLS IK → ㄴ자 복귀·완료 판정

기본 `guarded` 모드는 11초 안에 팔이 안정되지 않으면 중단합니다.
`--mode reference`는 시뮬에서만 원본의 11초 강제 전환을 재현합니다.
30초 전체 제한, 비정상 수치, 물체 낙하, 실험용 속도 중단 기준(3 rad/s)을 적용합니다.
이 속도는 실물 안전 한계가 아니며 원본 모터의 1 rad/s 제한도 완전히 재현하지 못했습니다.
MuJoCo의 마찰 모델은 PhysX의 정지·동마찰 및 결합 규칙과 같지 않습니다.

**정책은 연결됐지만 MuJoCo 파지 성공은 아직 검증되지 않았습니다.**
최초 edge10 실행에서는 접근·삽입 후 두 번째 정책까지 실행됐지만 완료 기준을 달성하지 못해 30초에 중단됐습니다.
학습 원래 배치에서도 같은 완료 실패가 있어 접촉·모터 모델과 원본 동작 대조가 필요합니다.
단순히 프로그램이 끝난 것을 파지 성공으로 표시하지 않으며 FAULT는 종료 코드 1을 반환합니다.

로그는 `runs/policy_날짜_시간/`에 저장됩니다.
`manifest.json`에는 파일 해시와 버전, `summary.json`에는 단계·실패 이유·정책 호출 횟수,
`trace.npz`에는 관절·목표·힘·큐브·관측·행동을 저장합니다.
`input_pad_n`과 `input_adc`는 추론 직전에 읽은 정확한 센서값입니다.
`observation[:,37:43]`이 `input_adc`와 같은지 검증했습니다.
`pad_n`, `pad_policy_n`은 해당 물리 스텝이 끝난 후 값이므로 입력값과 시점이 다릅니다.
실행 터미널에도 매초 및 접촉 시작/종료 때 PAD N과 ADC가 표시됩니다.
압력은 실물 센서 수신값이 아닌 MuJoCo 접촉력에서 생성한 가상 값입니다.
관측은 길이 54로 저장하며 잡기 정책 관측의 뒤 11개는 저장용 0 padding입니다.
수치 단위 테스트: `.venv/bin/python -m unittest discover -s tests -p test_policy.py -v`

## 책상·큐브 장면 실행

실측 높이 73.5 cm에 대해 거리 해석 두 가지를 제공합니다.
기본 CLI 배치는 `center10`입니다. 원본 정책의 배치는 `--layout reference`로 별도 유지합니다.

```sh
# A: 베이스 중심에서 책상 모서리까지 10 cm
.venv/bin/mjpython -m sim_validation.play_scene --layout center10
# B: 모델 받침대 끝에서 책상 모서리까지 10 cm
.venv/bin/mjpython -m sim_validation.play_scene --layout edge10
```

노란 `gap_10cm` 선은 해당 배치에서 10 cm를 재는 양 끝점을 나타냅니다.
`edge10`은 base_link 외형의 책상 방향 끝(베이스 중심에서 12.5 cm)을 사용하므로
중심에서 책상까지는 22.5 cm입니다. 실제 받침대 크기가 다르면 수정해야 합니다.
큐브 기준점은 각 책상 모서리에서 안쪽으로 6 cm, 옆 방향은 기존 26 cm를 유지합니다.
기준 큐브 중심 높이는 75.5 cm입니다. 초록점 높이는 책상보다 10 cm 높게 표시합니다.
이 변경은 장면 배치에만 적용되며 기존 모델 입력·경로의 실측 배치 보정은 아직 구현하지 않았습니다.

```sh
.venv/bin/mjpython -m sim_validation.play_scene
# 잡기 직전 자세에서 정적으로 시작
.venv/bin/mjpython -m sim_validation.play_scene --pose pregrasp
# 큐브를 기준점에서 책상 안쪽으로 3 cm 이동
.venv/bin/mjpython -m sim_validation.play_scene --dx-cm -3
# 특정 관절에 절대 목표 각도 5° 설정 (창에서 SPACE를 눌러 실행)
.venv/bin/mjpython -m sim_validation.play_scene --joint shoulder_yaw_to_upper_arm --degrees 5
```

- 처음에는 일시 정지. SPACE: 물리 실행/정지
- 0/1/2: 늘어진 자세/사전 파지/ㄴ자 목표 선택. 실행 중에는 목표 변화율 제한을 거쳐 이동
- R: 시작 자세와 큐브 위치로 초기화 후 정지
- C: 외형/충돌 형상 표시 전환
- 빨간 큐브, 갈색 책상, 초록점 `placement_0_0`: 기준점의 수평 위치
- 초록점은 가려지지 않도록 책상 위 10 cm에 표시하며 실제 큐브 중심은 책상 위 2 cm

각도 명령은 deg 단위 절대 관절 목표이며 관절 한계를 벗어나면 거부합니다.
사전 정의 자세 간 이동은 경로 계획이 아닌 직접 목표 변경입니다.
목표 변화율은 0.3 rad/s지만 접촉 등에 의한 실제 속도를 보장하는 제한은 아닙니다.
정책·접근 경로·압력 센서·가상 ADC는 아직 연결하지 않았습니다.
자기 충돌은 원본 Isaac처럼 끄고 책상·큐브와의 접촉은 유지합니다.
장면의 PD와 물리는 탐색용이며 전체 체인의 안전 검증 결과가 아닙니다.
창 없이 검사: `.venv/bin/python -m sim_validation.play_scene --headless-steps 120`

## 구조

- `tests/`: 로봇 모델 및 제어 검증
- `models/`: 제공된 `URDF_1006 (1).zip`에서 가져온 URDF 및 메시 32개
- `JaxIsaacLab/`: 별도로 복제한 정책·시뮬레이션 참고 저장소 (커밋 제외)

## URDF 정적 검사

Python 표준 라이브러리만 사용합니다. 모터 연결이나 명령 전송은 하지 않습니다.

```sh
cd /Users/jungwoo0719/sim-to-real
python3 -m unittest discover -s tests -v
```

기본적으로 저장소의 `models/JAXV2_Assembly_description/urdf/JAXV2_Assembly.urdf`를 검사합니다.
다른 모델을 검사할 때는 `ROBOT_URDF`를 해당 URDF 경로로 지정하세요.
메시의 `package://` 경로는 지정한 파일 경로, URDF 옆 파일,
패키지 루트, URDF 옆 `meshes/JAXV2_Assembly/`에서도 찾습니다.

현재 검사는 관절 이름·한계, 질량·관성의 기본 유효성, 메시 파일 존재 여부를 확인합니다.
누락된 충돌 메시도 실패로 보고합니다. 이 검사를 통과해도 물리 시뮬레이션이나 실물 안전 검증이 끝난 것은 아닙니다.

제공된 ZIP의 URDF는 CANWeb의 URDF와 바이트 단위로 동일합니다.
충돌 STL 16개와 외형 DAE 16개를 확보했으며 정적 검사 3개가 통과했습니다.
다음 단계는 MuJoCo 로딩 → 제어 제한 및 정책 동작 검증입니다.

## MuJoCo 검사

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

`sim_validation/model.py`는 원본 URDF를 변경하지 않고 로봇을 로드합니다.
검사 기본값은 충돌 STL만 사용하고, `visuals=True`는 DAE 외형도 변환해서 표시합니다.
DAE의 scene 변환과 URDF origin·scale을 유지하며 원본 충돌 스케일도 유지합니다.
현재 색상은 URDF material을 사용하고 DAE 내부의 부품별 재질은 가져오지 않습니다.
이 단계에는 책상·큐브·정책·실물 모터 연결이 없습니다.

- 컴파일된 14관절 이름과 각도 제한 확인
- 초기 자세에서 1 mm보다 깊은 자기 충돌 검사 (검출 시 실패)
- 관절마다 2° 이내 목표 이동: 유한값, 속도 1 rad/s 미만, 각도 제한 및 목표 도달 확인

관절 이동 검사는 중력과 충돌을 끄고 computed-torque 드라이버로 수행합니다.
실제 서보 PD나 정책 성능을 검증하는 검사가 아닙니다.
MuJoCo 관절 순서는 Isaac 순서와 다를 수 있어 이름으로 대응합니다.

참고: https://mujoco.readthedocs.io/en/latest/XMLreference.html

## 가상 하드웨어 실행

모터 대신 MuJoCo 관절을 읽고 목표각을 전달하는 60 Hz 루프입니다.
어깨 요 관절을 최대 2° 움직였다 되돌리며 CSV를 저장합니다.

```sh
# 화면 없이 실행
.venv/bin/python -m sim_validation.dry_run --ignore-contacts
# macOS에서 화면 표시
.venv/bin/mjpython -m sim_validation.dry_run --viewer --ignore-contacts
```

`--ignore-contacts`가 없으면 현재 초기 자세의 형상 겹침 때문에 움직이기 전에 중단합니다.
이 옵션은 어댑터 점검용이며 충돌 안전을 검증하지 않습니다.
이 실행기는 학습 정책, 큐브, 압력 센서를 아직 연결하지 않았습니다.
모델의 무구동 관절에 가상 PD·중력 보상, armature 0.1, damping 2를 추가하고
600 Hz 물리로 계산하므로 Isaac/실물 모터 재현 실험은 아닙니다.
목표 변화율 0.1 rad/s, 토크 ±10 N·m를 제한하며,
실제 시뮬 속도 1 rad/s 초과·관절 한계 초과·비정상 수치는 실행을 중단합니다.
기록: `runs/dry_run.csv` (Git 제외).

## 외형·자세 확인 (움직임 없음)

```sh
.venv/bin/mjpython -m sim_validation.inspect_model --pose zero
.venv/bin/mjpython -m sim_validation.inspect_model --pose pregrasp
.venv/bin/mjpython -m sim_validation.inspect_model --pose return
# 동일 자세의 충돌 형상만 보기
.venv/bin/mjpython -m sim_validation.inspect_model --pose zero --collision
```

외형은 viewer group 1, 충돌 형상은 group 0입니다. 기본은 외형만 표시합니다.
viewer의 Rendering → Geom groups에서 두 그룹을 켜 겹쳐 볼 수 있습니다.
정적 자세를 표시하며 정책·물리 적분·모터 통신은 실행하지 않습니다.
베이스는 URDF 원점이며 README의 시뮬 월드 배치는 아직 적용하지 않았습니다.

변환 검사:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_model_conversion.py' -v
```

URDF에서 별도로 계산한 FK와 MuJoCo 링크 좌표를 세 자세에서 비교합니다.
외형 추가 전후 질량과 접촉 결과가 같고, 외형이 충돌 계산에 들어가지 않는지도 검사합니다.
조사 결과는 [MODEL_VALIDATION.md](MODEL_VALIDATION.md)에 있습니다.
