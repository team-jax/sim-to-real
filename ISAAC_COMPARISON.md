# 원본 Isaac 설정과 MuJoCo 비교

확인일: 2026-11-05. 공개 main 문서와 배포 브랜치의 실제 제어 코드를 읽어 대조했다.

## 마찰값에 대한 결론

원본 재현 기준은 큐브·로봇 링크 **정지 0.4 / 동 0.35**다.
`GL_MU=0.4,0.35`가 큐브의 재질과 기본 시뮬 재질(로봇 링크)을 함께 바꾼다.
코드의 PLA–PLA 약 0.3–0.5라는 주석은 실측 증명이 아니다.
따라서 이 저장소를 근거로 특정 실제 출력물의 마찰계수를 확정할 수는 없다.

책상은 정지 1.2 / 동 1.0을 별도로 유지한다.
문서의 PhysX 평균 결합이 적용된다고 가정하면 큐브–책상은 정지 0.8 / 동 0.675다.
이는 명시된 값으로 계산한 결과이며 실제 PhysX 실행 중 조회한 값은 아니다.

학습 랜덤화는 큐브와 로봇 양쪽에 같은 정지마찰을 0.25–0.5에서 뽑아 적용한다.
동마찰은 그보다 0.05 낮다. 모델 요약은 원래 크기·배치에서 마찰 0.4일 때
12초 유지 99.0%, 0.3일 때 92.7%라고 보고한다. 이 평가를 직접 재실행하지는 않았다.

최초 MuJoCo 구현은 큐브–손·책상 접촉을 모두 0.4로 고정했다.
이후 사용자 요청에 따라 **동마찰 기준의 평균 결합 근사**로 수정했다.
현재 명시적 접촉쌍은 큐브–로봇 0.35, 큐브–책상 및 로봇–책상 0.675다.
MuJoCo의 단일 sliding friction으로 PhysX의 정지·동마찰 전환을 동시에 맞출 수 없으므로
완전한 엔진 재현이 아닌 동마찰 기준 근사로 표시한다. 큐브 크기·무게는 4 cm·50 g을 유지했다.
로컬 실제 table–cube 접촉의 `data.contact[].friction`을 조회해 0.675를 검증했다.

## 발견하고 수정한 정책 입력 불일치

원본 `JaxV2FrontGraspEnv.pad_forces()`는 엄지 두 패드에 전체 접촉력 크기를 사용한다.
정책 관측과 완료 판정은 이 재정의를 사용한다.
반면 `_balanced_close()`는 `JaxV2PadSensorGraspEnv.pad_forces(self)`를 직접 호출하여
엄지를 포함한 모든 패드의 수직 성분으로 힘 제어한다.

기존 MuJoCo 구현은 관측·완료 판정에도 모두 수직 성분을 넣고 있었다.
`deploy/observations.py`와 완료 판정을 수정하여 다음을 구분했다.

- 정책 관측·완료 판정: 손가락 1·2 수직 성분, 엄지 두 패드 전체 힘 크기
- 닫기 힘 제어: 모두 패드 수직 성분
- 로그 `pad_n`: 수직 성분, `pad_policy_n`: 정책용 엄지 크기를 적용한 값

옆면 접촉인 엄지는 수직 성분 0 N이면서 크기는 5 N일 수 있다.
회귀 테스트로 두 경로를 구별하고 ONNX/TorchScript 출력 일치 검사도 통과했다.
이 수정으로 파지 실패 원인이 모두 해결됐다는 의미는 아니다.

## 재실행 결과

엄지 관측 수정 직후에는 사용자 조건(한 변 4 cm, 50 g, 마찰 0.4, 책상 높이 73.5 cm, edge10)을 유지했다.
수정 후 실행에서도 접근 안정화·삽입 후 잡기 정책이 실행됐으나,
4 cm 상승 유지 완료 조건을 달성하지 못하고 30초 제한에서 중단됐다.
정책 관련 테스트 6개는 통과했다. 최초 실험 로그:
`runs/policy_20261105_202355_198859/summary.json` (Git 제외).

원본 물체는 한 변 5.5 cm·55 g이고 기존 조회표도 그 조건에서 만들어졌다.
4 cm 물체에 맞는 경로·접촉 재검증과 모터·마찰 모델 대조가 여전히 필요하다.
무게 50 g은 문서의 40–100 g 범위 안이지만 크기 4 cm는 제시된 크기 랜덤화 근거가 없다.

## 마찰 수정 후 압력 전달 검증

`runs/policy_20261105_203241_396011/`에서 30초를 재실행했다.
여전히 완료하지 못해 제한 시간에 중단됐으며 압력은 1800프레임 중 7프레임에서 0보다 컸다.
10.1667초의 엄지 끝마디 정책용 힘은 1.3538 N, ADC는 0.5083이었다.
저장한 모든 프레임에서 모델 관측 37–42번과 추론 직전 `input_adc`가 절대오차 3e-8 이내에서 일치했다.
따라서 압력 생성과 정책 입력 연결은 확인됐으나 안정적인 파지 접촉이 지속되지는 않는다.
장면 테스트 6개와 정책 테스트 6개가 통과했다.

## 근거

- [배포 README 6·10–12절](https://github.com/team-jax/JaxIsaacLab/blob/main/artifacts/deploy_2026-11-05/README.md)
- [마찰 설정과 랜덤화 코드](https://github.com/team-jax/JaxIsaacLab/blob/jax-one-arm-grasp-sim2real/source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_grasp_lift.py#L321)
- [엄지 힘 크기 재정의](https://github.com/team-jax/JaxIsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_grasp.py#L61)
- [원본 힘 제어](https://github.com/team-jax/JaxIsaacLab/blob/jax-one-arm-grasp-sim2real/source/isaaclab_tasks/isaaclab_tasks/direct/one_arm_jax_minimal/v2_front_grasp_close.py#L300)
- [모델 평가 요약](https://github.com/team-jax/JaxIsaacLab/blob/jax-one-arm-grasp-sim2real/artifacts/history/MODEL_SUMMARY_2026-11-05.md)
