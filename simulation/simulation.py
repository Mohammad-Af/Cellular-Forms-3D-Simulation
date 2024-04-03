import math
from cellular.cell import Cell
import random
import numpy as np
from main import render


def initialize(samples=1, scale_factor=2):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x / scale_factor, y / scale_factor, z / scale_factor))

    return points


def non_neighbor_cells(cells, cell):
    res = []
    for c in cells:
        if c not in cell.neighbors:
            res.append(c)
    return res


def most_near_cells(cells, cell, number):
    res = cells.copy()
    res.sort(key=lambda x: np.linalg.norm(x.coordinate - cell.coordinate))
    return res[:number]


def near_cells(cells, cell, radius_of_influence):
    res = []
    for c in cells:
        if cell.distance(c) < radius_of_influence:
            res.append(c)
    return res


def spring_target(cells, link_rest_length):
    res = []
    for cell in cells:
        s = np.array([0.0, 0.0, 0.0])
        for neighbor in cell.neighbors:
            s += neighbor.coordinate + link_rest_length * (cell.coordinate - neighbor.coordinate)
        res.append(s / len(cell.neighbors))
    return res


def planar_target(cells):
    res = []
    for cell in cells:
        s = np.array([0.0, 0.0, 0.0])
        for neighbor in cell.neighbors:
            s += neighbor.coordinate
        res.append(s / len(cell.neighbors))
    return res


def bulge_target(cells, link_rest_length):
    res = []
    for cell in cells:
        N = cell.coordinate / np.linalg.norm(cell.coordinate)
        bulge_dist = np.array([0.0, 0.0, 0.0])
        for neighbor in cell.neighbors:
            D = neighbor.coordinate - cell.coordinate
            if np.linalg.norm(D) ** 2 < link_rest_length ** 2:
                dotN = D.dot(N)
                # print(link_rest_length ** 2 - np.linalg.norm(neighbor.coordinate - cell.coordinate) ** 2 + dotN ** 2)
                bulge_dist += math.sqrt(
                    link_rest_length ** 2 - np.dot(D, D) + dotN ** 2) + dotN
        bulge_dist /= len(cell.neighbors)
        res.append(cell.coordinate + bulge_dist * N)
    return res


def collision_offset(cells, repulsion_strength, radius_of_influence):
    res = []
    for cell in cells:
        A = non_neighbor_cells(near_cells(cells, cell, radius_of_influence), cell)
        s = np.array([0.0, 0.0, 0.0])
        for c in A:
            if c != cell:
                s += ((radius_of_influence ** 2 - np.linalg.norm(
                    cell.coordinate - c.coordinate) ** 2) / radius_of_influence ** 2) * (
                             cell.coordinate - c.coordinate)
        s *= repulsion_strength
        res.append(s)
    return res


def split_neighbors(cell):
    # print(len(cell.neighbors))
    n1, n2 = random.sample(cell.neighbors, 2)
    v1 = n1.coordinate - cell.coordinate
    v2 = n2.coordinate - cell.coordinate
    cleavage_neighbors = [n1, n2]
    normal = np.cross(v1, v2)
    new_neighbors = []
    for neighbor in cell.neighbors:
        # if neighbor != n1 and neighbor != n2:
        if int(normal.dot(neighbor.coordinate - cell.coordinate)) < 0:
            new_neighbors.append(neighbor)
    return cleavage_neighbors, new_neighbors


def split(cells, split_threshold, params):
    for cell in cells:
        if cell.food > split_threshold:
            cell.food /= 2
            child = Cell(np.array([0.0, 0.0, 0.0]), radius=0.1, food=cell.food, neighbors=[], color=2)
            cleavage_neighbors, new_neighbors = split_neighbors(cell)
            for neighbor in new_neighbors:
                child.coordinate += neighbor.coordinate
                child.neighbors.append(neighbor)
                neighbor.neighbors.append(child)

                cell.neighbors.remove(neighbor)
                neighbor.neighbors.remove(cell)

            for neighbor in cleavage_neighbors:
                child.coordinate += neighbor.coordinate
                child.neighbors.append(neighbor)
                neighbor.neighbors.append(child)

            # update parent position
            # for neighbor in cell.neighbors:
            #     cell.coordinate += neighbor.coordinate
            # cell.coordinate /= (len(cell.neighbors) + 1.0)

            # Add extra neighbors
            for c in non_neighbor_cells(most_near_cells(cells, child, number=params['most_near_t']), child):
                if c is not child:
                    child.neighbors.append(c)
                    c.neighbors.append(child)
                    child.coordinate += c.coordinate

            child.neighbors.append(cell)
            cell.neighbors.append(child)

            # update child position
            child.coordinate += cell.coordinate
            child.coordinate /= len(child.neighbors)

            cells.append(child)

            # scale = move(cells, params)
            # save(cells, scale)

    return cells


def feed(cells, food_level):
    # TODO
    for cell in cells:
        cell.food += food_level + random.random()


def move(cells, params):
    link_rest_length = params['link_rest_length']
    repulsion_strength = params['repulsion_strength']
    radius_of_influence = params['radius_of_influence']
    spring_factor = params['spring_factor']
    bulge_factor = params['bulge_factor']
    planar_factor = params['planar_factor']
    spring = spring_target(cells, link_rest_length)
    planar = planar_target(cells)
    bulge = bulge_target(cells, link_rest_length)
    collision = collision_offset(cells, repulsion_strength, radius_of_influence)

    mean_coordinate_change = np.array([0.0, 0.0, 0.0])
    for i, cell in enumerate(cells):
        change = spring_factor * (spring[i] - cell.coordinate) + planar_factor * (
                planar[i] - cell.coordinate) + bulge_factor * (bulge[i] - cell.coordinate) + collision[i]
        cell.coordinate += change
        mean_coordinate_change += change

    mean_coordinate_change /= len(cells)
    mean_coordinate = np.array([0.0, 0.0, 0.0])
    for cell in cells:
        cell.coordinate -= mean_coordinate_change
        mean_coordinate += cell.coordinate

    mean_coordinate /= len(cells)
    min_x = 10000
    max_x = -10000
    min_y = 10000
    max_y = -10000
    for cell in cells:
        cell.coordinate -= mean_coordinate
        min_x = min(min_x, cell.coordinate[0] - (0.1 + cell.food / 10000.0))
        min_y = min(min_y, cell.coordinate[1] - (0.1 + cell.food / 10000.0))
        max_x = max(max_x, cell.coordinate[0] + (0.1 + cell.food / 10000.0))
        max_y = max(max_y, cell.coordinate[1] + (0.1 + cell.food / 10000.0))

    return max(max_x - min_x, max_y - min_y)


def simulate(cells, params):
    split_threshold = params['split_threshold']

    feed(cells, params['food_level'])
    cells = split(cells, split_threshold, params)
    scale = move(cells, params)
    save(cells, scale, params['folder_name'])


SAVE_NUM = 0


def save(cells, scale, folder_name):
    global SAVE_NUM
    f = open(f"./res/output/{folder_name}/coords_{SAVE_NUM}.txt", "w")
    f.write(str(scale))
    f.close()
    f = open(f"./res/output/{folder_name}/{SAVE_NUM}.txt", "w")

    for cell in cells:
        f.write(str(cell.coordinate[0]))
        f.write(" ")
        f.write(str(cell.coordinate[1]))
        f.write(" ")
        f.write(str(cell.coordinate[2]))
        f.write(" ")
        f.write(str(cell.food))
        f.write(" ")
        f.write(str(cell.color))
        f.write("\n")
    f.close()
    SAVE_NUM += 1


def main(folder_name):
    params = {
        'folder_name': folder_name,
        "iterations": 300,
        "n": 25,

        "food_level": 10.0,
        "most_near_t": 6,

        "split_threshold": 1000.0,
        "link_rest_length": "",

        "spring_factor": 0.07,
        "planar_factor": 0.05,
        "bulge_factor": 0.08,

        "repulsion_strength": 0.09,
        "radius_of_influence": ""
    }

    cells = []
    initial_coordinates = initialize(params['n'], scale_factor=5)
    for i in range(params['n']):
        cells.append(Cell(np.array(initial_coordinates[i]), radius=0.1, food=1, neighbors=[]))

    lrl = 0.0
    size = 0.0
    for cell in cells:
        for c in most_near_cells(cells, cell, params['most_near_t']):
            if c is not cell:
                cell.add_neighbor(c)
                lrl += np.linalg.norm(cell.coordinate - c.coordinate)
                size += 1

    lrl /= size
    params['link_rest_length'] = lrl
    params['radius_of_influence'] = random.uniform(lrl, 5 * lrl)

    for i in range(params['iterations']):
        simulate(cells, params)
        print('\r', "{:3.0f}%".format((float(i) / params['iterations']) * 100), end='')

    print(SAVE_NUM)


for i in range(10):
    SAVE_NUM = 0
    main(str(i))
