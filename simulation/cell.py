import math


class Cell:

    def __init__(self, coordinate, radius, food, neighbors, color=1):
        self.coordinate = coordinate
        self.radius = radius
        self.food = food
        self.neighbors = neighbors
        self.color = color

    def distance(self, cell):
        xs, ys, zs = self.coordinate
        xe, ye, ze = cell.coordinate
        return math.sqrt((xe - xs) ** 2 + (ye - ys) ** 2 + (ze - zs) ** 2)

    def add_neighbor(self, cell):
        self.neighbors.append(cell)
        # cell.neighbors.append(self)

