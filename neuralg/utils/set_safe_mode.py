def set_safe_mode(safe_mode=False):
    """ Enable running checks on inputs and outputs (e.g. eigval > 1e16 , input NaN etc.)

    Args:
        safe_mode (bool, optional): If true, checks are run on input and output. Defaults to False.
    """

    # Maybe should be a global variable?

