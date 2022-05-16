class SafeMode:
    """Wrapper for module safe mode variable"""

    def __init__(self):
        self.mode = None

    def set_mode(self, mode):
        """Set the safe mode

        Args:
            mode (bool): If True, the module is in safe mode and inputs/outputs will be checked
        """
        self.mode = mode
