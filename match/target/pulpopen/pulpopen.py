from match.target.target import MatchTarget
from match.target.pulpopen.cluster import ClusterModule

class PulpOpen(MatchTarget):
    def __init__(self):
        super(PulpOpen,self).__init__([
            ClusterModule(),
            ],name="pulpopen")
