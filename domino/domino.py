# Domino class with comparisons and other niceties.

class Domino:
    def __init__(self, a, b):
        self.value = (a,b) if a<b else (b,a)
        self.hash = hash(self.value)
        self.pip_val = a+b

    def fits(self, other):
        return (self.value[0] == other.value[0] or
                self.value[0] == other.value[1] or
                self.value[1] == other.value[0] or
                self.value[1] == other.value[1])

    def fits_val(self, other_val):
        return (True if other_val is None else 
                (self.value[0] == other_val or 
                 self.value[1] == other_val))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if other is None:
            return False
        if type(other) is tuple:
            return other == self.value
        return self.value == other.value

    def __lt__(self, other):
        return self.pip_val < other.pip_val
    
    def __le__(self, other):
        return self.pip_val <= other.pip_val 
    
    def __gt__(self, other):
        return not (self <= other)
    
    def __ge__(self, other):
        return not (self < other)
    
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)
    
    def __getitem__(self, key):
        return self.value[key]