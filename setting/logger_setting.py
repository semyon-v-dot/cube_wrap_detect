from .psr_qr_base_setting import PSR_QR_base_setting


class Logger_setting(PSR_QR_base_setting):
    name_project: str
    folder_log: str
    path_log_setting: str
    sys_log_address: str
    sys_log_port: str

    class Config:
        env_file = 'setting/test/logging.env'