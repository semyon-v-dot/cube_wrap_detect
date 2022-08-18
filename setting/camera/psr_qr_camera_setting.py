from ..psr_qr_base_setting import PSR_QR_base_setting
from typing import Optional, Union, List
from pydantic import validator



class PSR_QR_camera_setting(PSR_QR_base_setting):
    name_camera: str
    path_video: Union[str, List[str]]

    left: Optional[int]
    right: Optional[int]
    top: Optional[int]
    bottom: Optional[int]


    @validator("path_video")
    def validate_path_video(cls, v):
        if v[:4] == "rtsp":
            return v
        res = []
        for path_video in v.split("\n"):
            if path_video != "":
                res.append(path_video)
        return res