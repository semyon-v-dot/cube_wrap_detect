from .backend_setting import Backend_setting


class Backend_line_1_setting(Backend_setting):
    send_event: bool
    backend_url: str
    
    class Config:
        env_file = 'setting/test/backend_line_1.env'