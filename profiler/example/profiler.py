import sys
import json
from pathlib import Path


class Profile:
    profile_path = Path(__file__).parent / "profile.json"

    def __init__(self, group="Single-StateVector-Circuit"):

        with self.profile_path.open() as f:
            profile = json.load(f)

        if(group not in profile):
            print(f"Profile init: No data about {group}")
            sys.exit(1)
        self.group = group
        self.data = profile[group]
        


    def get_fast_space(self, nqubits: int, precision: str) -> str:
        """
        Get fast execution space based on given nqubits.
        If given 'nqubits' is out of index, get endpoint recommended execution space.
        (e.g. profile data(4~21): if given 'nqubits'=3, use 'nqubits'=4 data and if given 'nqubits'=30, use 'nqubits'=21 data)
        """

        if(precision not in self.data):
            print(f"Profile get_fast_space: No data about {self.group}-{precision}")
            sys.exit(1)

        if(nqubits < 1):
            print("Profile get_fast_space: nqubits must be 1 or more")
            sys.exit(1)


        min_nqubits = self.data[precision]["min_nqubits"]
        max_nqubits = self.data[precision]["max_nqubits"]
        
        if(nqubits < min_nqubits):
            return self._get_recommended_space(min_nqubits, precision)
        
        if(nqubits > max_nqubits):
            return self._get_recommended_space(max_nqubits, precision)
        
        return self._get_recommended_space(nqubits, precision)


    def _get_recommended_space(self, nqubits: int, precision: str) -> str:
        
        space = self.data[precision]["nqubits"][f"{nqubits}"]["recommended"]
        if space.startswith("default"):
            space = "default"
        return space
