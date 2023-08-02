import numpy as np 

class AntColonySystem:
    def __init__(self, distance_matrix, num_ants, num_iterations, alpha, beta, rho, q0):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        self.best_path = None
        self.best_path_length = np.inf

    def run(self):
        for iteration in range(self.num_iterations):
            paths = self.construct_solutions()
            self.update_pheromone_trails(paths)
            self.update_best_path(paths)
            self.evaporate_pheromone()

        return self.best_path, self.best_path_length

    def construct_solutions(self):
        paths = []
        for ant in range(self.num_ants):
            path = self.construct_solution(ant)
            paths.append(path)
        return paths

    def construct_solution(self, ant):
        visited = np.zeros(self.num_cities, dtype=bool)
        path = []
        current_city = np.random.randint(self.num_cities)
        visited[current_city] = True
        path.append(current_city)

        for _ in range(self.num_cities - 1):
            next_city = self.choose_next_city(current_city, visited)
            visited[next_city] = True
            path.append(next_city)
            current_city = next_city

        return path

    def choose_next_city(self, current_city, visited):
        unvisited_cities = np.where(~visited)[0]
        pheromone_values = self.pheromone_matrix[current_city, unvisited_cities]
        heuristic_values = 1.0 / self.distance_matrix[current_city, unvisited_cities]
        probabilities = np.power(pheromone_values, self.alpha) * np.power(heuristic_values, self.beta)

        if np.random.rand() < self.q0:
            max_probability_index = np.argmax(probabilities)
            next_city = unvisited_cities[max_probability_index]
        else:
            probabilities = probabilities / np.sum(probabilities)
            next_city = np.random.choice(unvisited_cities, p=probabilities)

        return next_city

    def update_pheromone_trails(self, paths):
        delta_pheromone = np.zeros((self.num_cities, self.num_cities))

        for path in paths:
            path_length = self.calculate_path_length(path)
            for i in range(self.num_cities - 1):
                delta_pheromone[path[i], path[i + 1]] += 1.0 / path_length

        self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix + delta_pheromone

    def update_best_path(self, paths):
        for path in paths:
            path_length = self.calculate_path_length(path)
            if path_length < self.best_path_length:
                self.best_path_length = path_length
                self.best_path = path

    def evaporate_pheromone(self):
        self.pheromone_matrix *= (1 - self.rho)

    def calculate_path_length(self, path):
        length = 0
        for i in range(self.num_cities - 1):
            length += self.distance_matrix[path[i], path[i + 1]]
        length += self.distance_matrix[path[-1], path[0]]
        return length

# reading file 1
f1 = open("gr229.tsp", "r")
txt = f1.readline().split()

while(txt[0] != "EOF"):
    txt = (f1.readline()).split()

    if(txt[0] == "DIMENSION:"):
        n = int(txt[1])
        cities = []
        f1.readline()
        f1.readline()
        f1.readline()

        for i in range(n):
            txt = (f1.readline()).split()
            cities.append([float(txt[1]),float(txt[2])])

cities = np.array(cities)
num_cities = len(cities)

distance_matrix = np.zeros((num_cities, num_cities))

for i in range(num_cities):
    for j in range(num_cities):
        distance_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])


num_ants = 5
num_iterations = 10
alpha = 3.0
beta = 5.0
rho  = 0.5
q0 = 0.7

aco = AntColonySystem(distance_matrix, num_ants, num_iterations, alpha, beta, rho, q0)
best_path, best_path_length = aco.run()

print("Best Path:", best_path)
print("Best Path Length:", best_path_length)