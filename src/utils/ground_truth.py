import copy
import json
import os
import subprocess
import threading
import time
from typing import Dict, List, Union

import numpy as np
import requests
from alive_progress import alive_bar
from moviepy.editor import VideoFileClip
from PIL import Image


class FaceAPIThread(threading.Thread):  # pragma: no cover
    """
    Thread that runs face-api.js as a background server.
    """

    def __init__(self, port: int, logging: bool = False) -> None:
        """
        Create a thread that runs face-api.js

        :param port: The port to start the API on.
        :param logging: Whether to use logging or not.
        """
        super().__init__()
        self.port = port
        self.api = None
        self.logging = logging

    def run(self) -> None:
        """
        Thread run function that starts a subprocess with face-api.js
        that listens on localhost:self.port/emotions.
        """
        env = copy.deepcopy(os.environ)
        env["PORT"] = f"{self.port}"
        self.api = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=os.path.join("src", "utils", "faceapi"),
            stdout=subprocess.DEVNULL if not self.logging else None,
            env=env,
        )

    def stop(self):
        """
        Stop function that stops the subprocess and the API.
        """
        self.api.kill()


def experiment_ground_truth(video_file: str) -> None:  # pragma: no cover
    """
    Main function that creates ground truth for the experiments
    using face expression emotions from face-api.js.

    :param video_file: The video_file to compute the emotions for.
    """
    video = VideoFileClip(video_file)
    port = 6060 + int(os.path.basename(video_file)[:3])
    print(
        f"Running face-api.js to get ground truth for experiment {port-6060}."
    )
    emotions = _get_emotions(video, port)
    destination = os.path.join(
        "data",
        "ground_truth",
        f"{os.path.basename(video_file).split('.')[0]}_emotions.json",
    )
    with open(destination, "w") as dest_file:
        json.dump(emotions, dest_file)


def _get_emotions(
    video: VideoFileClip, port: int
) -> List[List[Union[str, Dict[str, float]]]]:  # pragma: no cover
    """
    Get emotions for timestamps in the video.

    :param video: The video file to get the emotions for.
    :param port: The port to run the FaceAPI on.
    :return: Emotions list in the format needed for the API.
    """

    # Start the API for emotions prediction
    api = FaceAPIThread(port)
    api.start()
    time.sleep(30)  # Give the API enough time to start
    counter = 0
    frames = video.iter_frames()
    emotion_list = []
    fps = 30
    this_second = 0
    this_second_values = []
    with alive_bar(
        int(video.fps * video.duration) // 30 + 1,
        title="Frame",
        force_tty=True,
    ) as bar:
        for frame in frames:
            if counter // fps > this_second:
                emotion_list.append([str(counter / fps), this_second_values])
                this_second += 1
                this_second_values = []
            if counter % 30 == 0:
                if frame.shape[0] == 1080:
                    frame = frame.reshape((540, 2, 960, 2, 3)).max(3).max(1)
                else:
                    image = Image.fromarray(np.uint8(frame), "RGB")
                    image = image.resize((960, 540))
                    frame = np.array(image)
                payload = json.dumps({"image": frame.tolist()})
                headers = {
                    "Content-type": "application/json",
                    "Accept": "text/plain",
                }
                response = requests.post(
                    f"http://localhost:{port}/emotions",
                    data=payload,
                    headers=headers,
                )
                assert response.status_code == 200
                emotions = json.loads(response.content)["message"]
                if isinstance(emotions, dict):
                    for key, value in emotions.items():
                        emotions[key] = f"{value:.5f}"
                this_second_values.append(emotions)
                bar()
            counter += 1
    api.stop()
    api.join()
    return emotion_list
