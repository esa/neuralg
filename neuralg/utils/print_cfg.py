from dotmap import DotMap


def print_cfg(cfg: DotMap):
    """Prints the config in a more readable way.
    Args:
        cfg (DotMap): Config to print.
    """

    idx = 0
    for key, value in cfg.items():
        if isinstance(value, list):
            print()
            print(f"{key}: {value}")
        elif isinstance(value, DotMap):
            print()
            print(f"{key}:")
            print_dotmap(value)
        else:
            if idx % 2 == 2:
                print(f"{key:<5}: {value:<3}| ")
            else:
                print(f"{key:<5}: {value:<3}| ", end="")
        idx += 1
    print()


def print_dotmap(dm: DotMap):
    idx = 0
    for key, value in dm.items():
        if isinstance(value, list):
            print()
            print(f"{key}: {value}")
        else:
            if idx % 3 == 2:
                print(f"{key:<1}: {value:<1} ")
            else:
                print(f"{key:<1}: {value:<1} | ", end="")
                idx += 1
