from pydantic import BaseSettings


class PSR_QR_base_setting(BaseSettings):
    def __str__(self) -> str:
        result: str = f"{self.__class__.__name__}\n"
        dict = self.dict()
        for keys in dict:
            result += f"\t{keys}: {dict[keys]}\n"
        return result