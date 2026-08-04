"""Transport- and framework-independent policy schema contracts."""

from .feature_contract import FeatureContract, FeatureContractError, FeatureDescriptor
from .mapping import (
    MappingError,
    cartesian_minimal_state,
    flexiv_inspire_action_mappings,
    joint_minimal_state,
    legacy_state38,
    native_action30_to_policy24,
    policy_action24_to_native30,
)
from .registry import ActionMappingRegistry, ActionTensorMapping
from .profiles import (
    CARTESIAN_MINIMAL_PROFILE,
    DUAL_ARM_LEROBOT_V1_PROFILE,
    JOINT_MINIMAL_PROFILE,
    PolicyProfile,
    get_profile,
)
from .rotation import (
    matrix_to_rotation6d,
    matrix_to_rotvec,
    quaternion_xyzw_to_matrix,
    rotation6d_to_matrix,
    rotvec_to_matrix,
)
from .schema import (
    ActionSchema,
    ChannelDescriptor,
    SchemaError,
    SystemSchema,
    TensorDescriptor,
)

__all__ = [
    "ActionSchema",
    "ActionMappingRegistry",
    "ActionTensorMapping",
    "CARTESIAN_MINIMAL_PROFILE",
    "ChannelDescriptor",
    "DUAL_ARM_LEROBOT_V1_PROFILE",
    "FeatureContract",
    "FeatureContractError",
    "FeatureDescriptor",
    "JOINT_MINIMAL_PROFILE",
    "MappingError",
    "PolicyProfile",
    "SchemaError",
    "SystemSchema",
    "TensorDescriptor",
    "cartesian_minimal_state",
    "flexiv_inspire_action_mappings",
    "get_profile",
    "joint_minimal_state",
    "legacy_state38",
    "matrix_to_rotation6d",
    "matrix_to_rotvec",
    "native_action30_to_policy24",
    "policy_action24_to_native30",
    "quaternion_xyzw_to_matrix",
    "rotation6d_to_matrix",
    "rotvec_to_matrix",
]
