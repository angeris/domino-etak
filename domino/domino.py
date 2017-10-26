# Domino class with comparisons and other niceties.

class Domino:
    def __init__(self, a, b):
        if a < b:
            self.value = (a,b)
        else:
            self.value = (b,a)
    def __eq__(self, other_domino):
        return self.value == other_domino.value
    def __lt__(self, other_domino):
        return self.value[0] + self.value[1] < other_domino.value[0] + other_domino.value[1] 
    def __le__(self, other_domino):
        return self.value[0] + self.value[1] <= other_domino.value[0] + other_domino.value[1] 
    def __gt__(self, other_domino):
        return self.value[0] + self.value[1] > other_domino.value[0] + other_domino.value[1] 
    def __ge__(self, other_domino):
        return self.value[0] + self.value[1] >= other_domino.value[0] + other_domino.value[1] 
    def __str__(self):
        return str(self.value)