from .controller_snmp import PSR_QR_controller_snmp


class Controller_drop(PSR_QR_controller_snmp):
    class Config:
        env_file = 'setting/test/controller_drop.env'