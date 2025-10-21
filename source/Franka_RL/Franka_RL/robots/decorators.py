from .factory import RobotFactory

def register_dexhand(dexhand_type):
    def decorator(cls):
        RobotFactory.register(dexhand_type, cls)
        return cls

    return decorator
