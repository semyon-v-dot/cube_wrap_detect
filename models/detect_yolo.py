import cv2 as cv
import numpy as np
import torch
from typing import List
from show import Annotator, colors
from .t_yolo import (Model_backend, check_img_size, letterbox, non_max_suppression,
                    scale_coords, select_device, xyxy2xywh)
from setting import Yolo_setting


class Box:
    def __init__(
        self,
        label: str,
        left: int,
        right: int,
        top: int,
        bottom: int,
        conf: float = 0
    ):
        self.label = label
        self.conf = conf
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.center_x = (left + right) // 2
        self.center_y = (top + bottom) // 2
        self.height = abs(bottom - top)
        self.width = abs(right - left)
        self.area = self.height * self.width


    def check_nesting_box(
        self,
        external_box
    ) -> bool:
        if external_box.left < self.center_x and \
                self.center_x < external_box.right and \
                external_box.top < self.center_y and \
                self.center_y < external_box.bottom:
            return True
        
        return False


class Detect_yolo:
    def __init__(
        self,
        yolo_setting: Yolo_setting,
        imgsz = (640, 640)
    ) -> None:
        self.device = select_device(yolo_setting.device)
        self.model = Model_backend(yolo_setting.weights, device=self.device)
        self.stride, self.names = self.model.stride, self.model.names
        self.imgsz = check_img_size(imgsz, s=self.stride)

    @torch.no_grad()
    def detect(
        self,
        img: np.ndarray
    ) -> List[Box]:
        
        im = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im)
        pred = non_max_suppression(pred, classes=None)
        answer = []
        
        for _, det in enumerate(pred):
            im0 = img.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=3)
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                n = 0
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    
                    c = int(cls)
                    label = (f'{self.names[c]}')
                    if 0.5 < float(f"{conf:.2f}") and n < 50:
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        n += 1
                        answer.append(
                            Box(
                                label=label,
                                conf=float(conf),
                                left=int(xyxy[0]),
                                right=int(xyxy[2]),
                                top=int(xyxy[1]),
                                bottom=int(xyxy[3])
                            )
                        )
                    # annotator.box_label(xyxy, label, color=colors(c, True))

            # if True:
            #     im0 = annotator.result()
            #     cv.imshow("frame", cv.resize(im0, (1000, 800)))
            #     cv.waitKey(1)
        
        return answer
