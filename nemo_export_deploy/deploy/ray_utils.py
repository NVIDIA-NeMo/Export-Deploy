import socket


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """
    Check if a given port is already in use.

    Args:
        port (int): The port number to check.

    Returns:
        bool: True if the port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True


def find_available_port(start_port: int, host: str = "0.0.0.0") -> int:
    """
    Find the next available port starting from a given port number.

    Args:
        start_port (int): The port number to start checking from.

    Returns:
        int: The first available port number found.
    """
    port = start_port
    while is_port_in_use(port, host):
        port += 1
    return port
