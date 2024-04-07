from termcolor import colored
import logging

logging.basicConfig(filename='exceptions.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class AdvancedCustomException(Exception):
    def __init__(self, original_exception, color="red"):
        self.original_exception = original_exception
        self.color = color
        self.message = str(original_exception)
        self.code = self.extract_code(original_exception)

    def extract_code(self, exception):
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            return exception.response.status_code
        return 500

    def __str__(self):
        color1 = "red"
        color2 = "green"
        color3 = "blue"
        
        part0 = colored("[Error ", color1)
        part1 = colored(f"{self.code}", color2)
        part2 = colored("]:", color1)
        part3 = colored(f" {self.message}", color3)

        return part0 + part1 + part2 + part3
