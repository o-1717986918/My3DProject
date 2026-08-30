from pathlib import Path

from my3d_rl import load_policy_contract


CONTRACT = Path(__file__).parents[1] / "contracts" / "kick_policy_v1.yaml"


def test_kick_policy_contract_is_internally_consistent():
    contract = load_policy_contract(CONTRACT)

    assert contract.policy_name == "kick_policy_v1"
    assert contract.frequency_hz == 50
    assert contract.action_size == 23
    assert contract.observation_size == 90
    assert contract.input_shape == (1, 90)
    assert contract.output_shape == (1, 23)


def test_joint_order_matches_rcssservermj_sensor_order():
    contract = load_policy_contract(CONTRACT)

    assert contract.joint_order[:3] == (
        "AAHead_yaw",
        "Head_pitch",
        "Left_Shoulder_Pitch",
    )
    assert contract.joint_order[-3:] == (
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    )
    assert contract.effector_order == (
        "he1",
        "he2",
        "lae1",
        "lae2",
        "lae3",
        "lae4",
        "rae1",
        "rae2",
        "rae3",
        "rae4",
        "te1",
        "lle1",
        "lle2",
        "lle3",
        "lle4",
        "lle5",
        "lle6",
        "rle1",
        "rle2",
        "rle3",
        "rle4",
        "rle5",
        "rle6",
    )
