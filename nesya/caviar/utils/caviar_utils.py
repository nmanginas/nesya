import os
import shutil
import requests

from rich.progress import track
from xml.etree import ElementTree


def cache_caviar_raw(chunk_size: int = 4096) -> str:
    response = requests.get(
        "https://users.iit.demokritos.gr/~nkatz/caviar_videos.zip", stream=True
    )

    zip_size = int(response.headers.get("content-length"))  # type: ignore
    cache_path = os.path.expanduser(os.path.join("~", ".cache", "caviar_raw"))

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    if not os.path.exists(zip_path := os.path.join(cache_path, "caviar.zip")):
        with open(os.path.join(cache_path, "caviar.zip"), "wb") as output_file:
            for data in track(
                response.iter_content(chunk_size),
                description="downloading raw",
                total=zip_size // chunk_size,
            ):
                output_file.write(data)
    else:
        if os.stat(zip_path).st_size != zip_size:
            print("Found archive but is is partial. Re-downloading")
            os.remove(zip_path)
            cache_caviar_raw()

    if not os.path.exists(extracted_path := os.path.join(cache_path, "caviar_videos")):
        shutil.unpack_archive(zip_path, cache_path)

    return extracted_path


def parse_caviar_dataset(root_dir: str, minimum_frames: int = 20):
    xml_files = list(filter(lambda x: x.endswith(".xml"), os.listdir(root_dir)))
    children = lambda x: list(iter(x))
    complex_event_data = {}
    for current_file in xml_files:
        xml_file = os.path.join(root_dir, current_file)
        root = ElementTree.parse(xml_file).getroot()
        video_file: str = root.attrib["name"] + ".mpg"
        frames = children(root)

        parsed_frames = []
        group_info = {}

        for i, frame in enumerate(frames):
            objectlist, grouplist = children(frame)
            frame_objects = {}
            for object in children(objectlist):
                orientation, box, *_, hypothesislist = children(object)
                (
                    orientation,
                    box_height,
                    box_width,
                    box_xcenter,
                    box_ycenter,
                    simple_event,
                ) = (
                    int(orientation.text),
                    int(box.attrib["h"]),
                    int(box.attrib["w"]),
                    int(box.attrib["xc"]),
                    int(box.attrib["yc"]),
                    children(children(hypothesislist)[0])[0].text,
                )
                frame_objects[int(object.attrib["id"])] = {
                    "orientation": orientation,
                    "box_height": box_height,
                    "box_width": box_width,
                    "box_xcenter": box_xcenter,
                    "box_ycenter": box_ycenter,
                    "simple_event": simple_event,
                }

            groups = children(grouplist)
            frame_groups = {}
            for group in groups:
                orientation, box, members, _, hypothesislist = children(group)
                (
                    orientation,
                    box_height,
                    box_width,
                    box_xcenter,
                    box_ycenter,
                    members,
                    complex_event,
                ) = (
                    int(orientation.text),
                    int(box.attrib["h"]),
                    int(box.attrib["w"]),
                    int(box.attrib["xc"]),
                    int(box.attrib["yc"]),
                    tuple(map(int, members.text.split(","))),
                    children(children(hypothesislist)[0])[-1].text,
                )
                if int(group.attrib["id"]) not in group_info:
                    group_info[int(group.attrib["id"])] = {
                        "members": members,
                        "start_frame": i,
                    }
                frame_groups[int(group.attrib["id"])] = {
                    "orientation": orientation,
                    "box_height": box_height,
                    "box_width": box_width,
                    "box_xcenter": box_xcenter,
                    "box_ycenter": box_ycenter,
                    "members": members,
                    "complex_event": complex_event,
                }

            parsed_frames.append({"objects": frame_objects, "groups": frame_groups})

        for group_id, info in group_info.items():
            start_tracking_frame = info["start_frame"]
            while start_tracking_frame:
                if all(
                    member in parsed_frames[start_tracking_frame]["objects"]
                    for member in info["members"]
                ):
                    start_tracking_frame -= 1
                    continue
                break

            end_tracking_frame = info["start_frame"]
            while end_tracking_frame < len(parsed_frames):
                if all(
                    member in parsed_frames[end_tracking_frame]["objects"]
                    for member in info["members"]
                ):
                    end_tracking_frame += 1
                    continue
                break

            history = []

            for i in range(start_tracking_frame + 1, end_tracking_frame):
                frame = parsed_frames[i]
                frame_objects = [
                    object
                    for object_id, object in frame["objects"].items()
                    if object_id in info["members"]
                ]

                if len(frame_objects) != 2:
                    raise RuntimeError(
                        "We always track one pair of people found {} objects".format(
                            len(frame_objects)
                        )
                    )

                left, right = frame_objects
                import math

                distance = math.dist(
                    (left["box_xcenter"], left["box_ycenter"]),
                    (
                        right["box_xcenter"],
                        right["box_ycenter"],
                    ),
                )

                history.append(
                    {
                        "frame_id": i,
                        "objects": frame_objects,
                        "distance": distance,
                        "complex_event": (
                            frame["groups"][group_id]["complex_event"]
                            if group_id in frame["groups"]
                            else "no_event"
                        ),
                        "complex_event_box_height": (
                            frame["groups"][group_id]["box_height"]
                            if group_id in frame["groups"]
                            else 0
                        ),
                        "complex_event_box_width": (
                            frame["groups"][group_id]["box_width"]
                            if group_id in frame["groups"]
                            else 0
                        ),
                        "complex_event_box_xcenter": (
                            frame["groups"][group_id]["box_xcenter"]
                            if group_id in frame["groups"]
                            else 0
                        ),
                        "complex_event_box_ycenter": (
                            frame["groups"][group_id]["box_ycenter"]
                            if group_id in frame["groups"]
                            else 0
                        ),
                    }
                )

            if len(history) > minimum_frames:
                complex_event_data[
                    "{}:{}".format(current_file.removesuffix(".xml"), group_id)
                ] = {
                    "xml_file": xml_file,
                    "history": history,
                    "video_file": video_file,
                    "frame_range": (start_tracking_frame, end_tracking_frame),
                }

    return complex_event_data
