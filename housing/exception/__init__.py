import os 
import sys

class HousingException(Exception):
    def __init__(self, error_massage:Exception,error_detail:sys):
        super().__init__(error_massage)
        self.error_massage=error_massage
    @staticmethod
    def get_detailed_erroe_message(error_massage:Exception,error_detail:sys)->str:
        """ error_message: Exception object
        error_detail: object of sys module """
        _,_,exec_tb=error_detail.exc_info()

        line_number=exec_tb.tb_frame.f_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_massage = f"Error occured inscript :[{file_name}] at line numbeer :[{line_number}] error message : [{error_massage}]"
        return error_massage

    def __str__(self):
        return self.erroe_massage
    def __repr__(self)->str:
        return HousingException.__name__.str()

