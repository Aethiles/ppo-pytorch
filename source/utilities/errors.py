class BaseError(Exception):
    def __init__(self, message):
        self.message = message


class ShapeError(BaseError):
    """
    An Exception used to indicate a wrong shape
    """
    pass
