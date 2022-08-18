from ..psr_qr_base_setting import PSR_QR_base_setting


"""
.1.3.6.1.4.1.51014.2.301.44544.470

на вход подаешь число, оно преобразуется в двоичное
1 - зеленый
2 - желтый 
3 - красный
4 - зуммер
5 - сброс ? 
узнать состояние выходов:
.1.3.6.1.4.1.51014.2.301.44544.468
используем выход 8

10.98.2.27
"""


class PSR_QR_controller_snmp(PSR_QR_base_setting):
    community_string: str
    ip_address_host: str
    flag_buzzer_signal: bool