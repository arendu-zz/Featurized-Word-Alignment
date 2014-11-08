__author__ = 'arenduchintala'
import random


class SGD(object):
    def __init__(self, data, gradeint, likelihood):
        self.data = data
        self.graient = gradeint
        self.likelihood = likelihood



    def train(self):
        random.shuffle(self.data)
        for d in self.data:
            pass
