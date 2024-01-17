from match.target.gap9 import Gap9
from match.target.target import DefaultMatchTarget, MatchTarget

TARGETS={
    "gap9":Gap9
}

def get_target(target_name:str=""):
    if target_name not in TARGETS:
        return DefaultMatchTarget()
    else:
        assert issubclass(TARGETS[target_name],MatchTarget)
        return TARGETS[target_name]()