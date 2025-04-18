import os
import sys


class HousingException(Exception):
    """Custom exception class for handling errors in the Housing project."""

    def __init__(self, error_message: Exception, error_detail: sys):
        """Initialize HousingException with a detailed error message.

        Args:
            error_message (Exception): The exception object containing the error details.
            error_detail (sys): The sys module object to extract traceback information.

        Attributes:
            error_message (str): Formatted error message with file, line numbers, and error details.
        """
        # Call the parent class (Exception) initializer with the error message
        super().__init__(error_message)
        # Generate and store the detailed error message
        self.error_message = HousingException.get_detailed_error_message(error_message=error_message,
                                                                        error_detail=error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        """Generate a detailed error message with traceback information.

        Args:
            error_message (Exception): The exception object containing the error details.
            error_detail (sys): The sys module object to extract traceback information.

        Returns:
            str: A formatted string containing the file name, line numbers, and error message.
        """
        # Extract traceback information: exception type, value, and traceback object
        _, _, exec_tb = error_detail.exc_info()
        # Get the line number of the exception block
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        # Get the line number of the try block where the error occurred
        try_block_line_number = exec_tb.tb_lineno
        # Get the file name where the error occurred
        file_name = exec_tb.tb_frame.f_code.co_filename
        # Format the detailed error message
        error_message = f"""
        Error occurred in script: 
        [ {file_name} ] at 
        try block line number: [{try_block_line_number}] and exception block line number: [{exception_block_line_number}] 
        error message: [{error_message}]
        """
        return error_message

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The detailed error message.
        """
        return self.error_message

    def __repr__(self) -> str:
        """Return the formal string representation of the exception.

        Returns:
            str: The name of the HousingException class.
        """
        return HousingException.__name__.str()