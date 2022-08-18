import argparse

import cv2
import numpy as np
import torch

from show import Annotator, colors

from t_yolo import (Model_backend, check_img_size, letterbox, non_max_suppression,
                  scale_coords, select_device, xyxy2xywh)


@torch.no_grad()
def run(
        weights,  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
):
    device = select_device(device)
    model = Model_backend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)
    
    path = "H:\\packet_tracker\\pretrain_images\\uchaly\\upper_2\\carbon_long\\542528-video.h264\\250.png"
    im0s = cv2.imread(path)
    im = letterbox(im0s, imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]

    pred = model(im, augment=augment)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):
        im0 = im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        annotator = Annotator(im0, line_width=line_thickness)
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            n = 0
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                print(xywh)
                print((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])))
                c = int(cls)
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                if 0.5 < float(f"{conf:.2f}") and n < 20:
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    n += 1
                # annotator.box_label(xyxy, label, color=colors(c, True))

        im0 = annotator.result()
        if view_img:
            cv2.imshow("frame", cv2.resize(im0, (1000, 800)))
            cv2.waitKey(0)
        # if save_img:
        #     cv2.imwrite(save_path, im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
