import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_relative_pose_actions_use_so3_shortest_rotation_and_round_trip():
    state = np.array([0.4, -0.1, 0.3, np.pi - 0.01, 0.0, 0.0, 0.02])
    actions = np.tile(state, (3, 1))
    actions[:, :3] += np.array([[0.001, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, 0.003]])
    actions[:, 3] = -np.pi + 0.01
    actions[:, 6] = [0.01, 0.02, 0.03]
    original = actions.copy()

    relative = _transforms.RelativePoseActions()({"state": state.copy(), "actions": actions})
    np.testing.assert_allclose(relative["actions"][:, :3], original[:, :3] - state[:3], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(relative["actions"][:, 3:6], axis=-1), 0.02, atol=1e-12)
    np.testing.assert_array_equal(relative["actions"][:, 6], original[:, 6])

    restored = _transforms.AbsolutePoseActions()(relative)
    np.testing.assert_allclose(restored["actions"][:, :3], original[:, :3], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_rotvec(restored["actions"][:, 3:6]).as_matrix(),
        Rotation.from_rotvec(original[:, 3:6]).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_array_equal(restored["actions"][:, 6], original[:, 6])


def test_relative_pose_actions_support_batched_statistics_shape():
    state = np.zeros((2, 7))
    actions = np.zeros((2, 4, 7))
    actions[0, :, 3] = -np.pi + 0.01
    state[0, 3] = np.pi - 0.01
    result = _transforms.RelativePoseActions()({"state": state, "actions": actions})["actions"]
    assert result.shape == (2, 4, 7)
    np.testing.assert_allclose(np.linalg.norm(result[0, :, 3:6], axis=-1), 0.02, atol=1e-12)


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
