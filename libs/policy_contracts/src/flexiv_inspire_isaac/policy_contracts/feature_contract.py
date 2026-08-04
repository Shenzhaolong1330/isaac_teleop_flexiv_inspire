"""Compile a LeRobot metadata document against a versioned policy profile."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .profiles import PolicyProfile


class FeatureContractError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureDescriptor:
    key: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...]


@dataclass(frozen=True)
class FeatureContract:
    fps: float
    robot_type: str
    features: tuple[FeatureDescriptor, ...]

    @classmethod
    def from_info(cls, value: str | Path | Mapping[str, Any]) -> "FeatureContract":
        if isinstance(value, (str, Path)):
            document = json.loads(Path(value).read_text(encoding="utf-8"))
        else:
            document = dict(value)
        try:
            raw_features = document["features"]
            fps = float(document["fps"])
            robot_type = str(document["robot_type"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FeatureContractError(f"invalid LeRobot info document: {exc}") from exc
        if fps <= 0.0 or not robot_type or not isinstance(raw_features, Mapping):
            raise FeatureContractError("info requires positive fps, robot_type and features")
        parsed: list[FeatureDescriptor] = []
        for key, raw in raw_features.items():
            if not isinstance(raw, Mapping):
                raise FeatureContractError(f"feature {key} must be a mapping")
            try:
                dtype = str(raw["dtype"])
                shape = tuple(int(item) for item in raw["shape"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FeatureContractError(f"feature {key} is invalid: {exc}") from exc
            raw_names = raw.get("names")
            names = () if raw_names is None else tuple(str(item) for item in raw_names)
            if not shape or any(item <= 0 for item in shape):
                raise FeatureContractError(f"feature {key} has invalid shape")
            if names and (len(shape) != 1 or len(names) != shape[0]):
                # Image axis labels describe axes rather than flattened elements.
                if dtype != "video":
                    raise FeatureContractError(f"feature {key} names do not match shape")
            parsed.append(FeatureDescriptor(str(key), dtype, shape, names))
        return cls(fps=fps, robot_type=robot_type, features=tuple(parsed))

    def feature(self, key: str) -> FeatureDescriptor:
        for feature in self.features:
            if feature.key == key:
                return feature
        raise FeatureContractError(f"missing feature: {key}")

    def validate_profile(self, profile: PolicyProfile) -> None:
        state = self.feature("observation.state")
        action = self.feature("action")
        if state.shape != (profile.state_dimension,) or state.names != profile.state_names:
            raise FeatureContractError(
                f"observation.state does not match profile {profile.profile_id}"
            )
        if action.shape != (profile.action_dimension,) or action.names != profile.action_names:
            raise FeatureContractError(f"action does not match profile {profile.profile_id}")
        observation_keys = {
            item.key for item in self.features if item.key.startswith("observation.")
        }
        expected = {"observation.state", *profile.image_keys}
        if observation_keys != expected:
            extras = sorted(observation_keys - expected)
            missing = sorted(expected - observation_keys)
            raise FeatureContractError(
                f"profile observation fields differ; missing={missing}, extras={extras}"
            )

    @property
    def ordered_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.features)
