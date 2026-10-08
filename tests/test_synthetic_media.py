"""The simulated picture's drawing and recognising agree; a take can be joined into a video."""
from __future__ import annotations

import cv2
import pytest

from shotagent import media
from shotagent.perception.synthetic import SimScene, SyntheticDetector, render
from shotagent.spatial import facing_from_keypoints, relative_position


def detect(scene: SimScene, quality: int = 70) -> dict:
    """Render -> JPEG compression (the same quality the browser uploads) -> decode -> detect."""
    image = media.decode_jpeg(media.encode_jpeg(render(scene), quality=quality))
    return {d.label: d for d in SyntheticDetector().detect(image)}


def test_empty_scene_has_only_the_table():
    assert set(detect(SimScene(bag=None))) == {"table"}
    assert detect(SimScene(table=False, bag=None)) == {}


@pytest.mark.parametrize("u", [0.1, 0.2, 0.5, 0.8, 0.9])
def test_bag_position_round_trips_through_pixels(u):
    found = detect(SimScene(bag=u))
    rel = relative_position(found["bag"].box, found["table"].box)
    assert rel.u == pytest.approx(u, abs=0.02)
    assert rel.vertical == "under"


def test_bag_on_the_table_is_not_under_it():
    found = detect(SimScene(bag=0.3, bag_under=False))
    assert relative_position(found["bag"].box, found["table"].box).vertical == "above"


@pytest.mark.parametrize("facing", ["left", "right", "front"])
@pytest.mark.parametrize("x", [0.15, 0.4, 0.7])
def test_person_and_facing_are_decoded(x, facing):
    found = detect(SimScene(bag=0.2, person=x, facing=facing))
    person = found["person"]
    assert person.box.cx == pytest.approx(x, abs=0.02)
    assert facing_from_keypoints(person.keypoints) == facing


def test_person_standing_right_over_the_bag_is_still_decoded():
    found = detect(SimScene(bag=0.3, person=0.4, facing="left"))    # person, bag and table overlap each other
    assert {"table", "bag", "person"} <= set(found)


def test_clip_frames_become_a_playable_mp4(tmp_path):
    for index in range(8):
        scene = SimScene(bag=0.2, person=0.3 + index * 0.05)
        (tmp_path / f"{index:03d}.jpg").write_bytes(media.encode_jpeg(render(scene)))
    video = media.encode_mp4(tmp_path, fps=3)

    assert video.read_bytes()[4:8] == b"ftyp"          # an mp4 container
    capture = cv2.VideoCapture(str(video))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 8


def test_encode_mp4_refuses_an_empty_clip(tmp_path):
    with pytest.raises(ValueError):
        media.encode_mp4(tmp_path, fps=3)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8\xff\xe0 truncated jpeg header"])
def test_decode_jpeg_rejects_anything_that_is_not_an_image(data):
    with pytest.raises(ValueError):
        media.decode_jpeg(data)
