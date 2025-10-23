class Utils:
    """
    Utility methods for printing to console with specific colors.
    """
    
    @staticmethod
    def print_color(text: str, color: str = 'white') -> None:
        """
        Print text to console with specific color.
        
        Parameters:
        -----------
        text : str
            Text to print
        color : str
            Color to print text with
            
        Returns:
        --------
        None
        """
        colors = {
            'white': '\033[0m',
            'red': '\033[31m',
            'green': '\033[32m',
            'blue': '\033[34m',
            'yellow': '\033[33m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
        }
        
        color_code = colors.get(color, '')
        print(f"{color_code}{text}{colors['white']}")
