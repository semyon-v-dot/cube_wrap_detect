from .psr_qr_base_setting import PSR_QR_base_setting


class Yolo_setting(PSR_QR_base_setting):
    weights: str
    device: str

    class Config:
        env_file = 'setting/test/yolo.env'