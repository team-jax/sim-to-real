from pathlib import Path
import numpy as np
import onnxruntime as ort

DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[1] / "JaxIsaacLab/artifacts/deploy_2026-11-05"


class Policy:
    def __init__(self, path, observations):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.observations = observations
        self.filtered = np.zeros(11)
        self.last_raw = np.zeros(11)

    def reset(self):
        self.filtered[:] = 0
        self.last_raw[:] = 0

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observations,) or not np.isfinite(observation).all():
            raise ValueError("Invalid policy observation")
        raw = self.session.run(None, {self.input: observation[None]})[0]
        if raw.shape != (1, 11) or not np.isfinite(raw).all():
            raise ValueError("Invalid policy output")
        self.last_raw = raw[0].copy()
        self.filtered = 0.25 * np.clip(raw[0], -1, 1) + 0.75 * self.filtered
        return self.filtered.copy()
