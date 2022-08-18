from .psr_qr_camera_setting import PSR_QR_camera_setting


class Magazin_2_setting(PSR_QR_camera_setting):
    class Config:
        env_file = 'setting/test/magazin_2.env'