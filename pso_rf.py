import numpy as np
from tabulate import tabulate
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Particle class
class Particle:
    """ Particle class """

    def __init__(self, fitness_func, lower_bound, upper_bound):

        self.position = np.random.uniform(lower_bound, upper_bound)
        self.velocity = np.zeros(len(lower_bound))
        self.best_position = np.copy(self.position)
        self.best_value = fitness_func(self.position)
        self.current_fitness = self.best_value

    def update_velocity(self, global_best_position, inertia, b1, b2):
        """ Update velocity """

        r1, r2 = np.random.rand(), np.random.rand()
        
        cognitive_velocity = b1 * r1 * (self.best_position - self.position)
        social_velocity = b2 * r2 * (global_best_position - self.position)

        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity
    
    def update_position(self, fitness_func, lower_bound, upper_bound):
        """ Update position """

        self.position += self.velocity
        # Apply bounds
        self.position = np.clip(self.position, lower_bound, upper_bound)

        current_fitness = fitness_func(self.position)
        if current_fitness < self.best_value:
            self.best_position = np.copy(self.position)
            self.best_value = current_fitness
        self.current_fitness = current_fitness

# Particle Swarm Optimization class
class PSO:
    """ Particle Swarm Optimization class """

    def __init__(self, fitness_func, lower_bound, upper_bound, num_particles = 20, num_iterations = 50, inertia = 0.8, b1 = 0.7, b2 = 1.2):
        
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if len(lower_bound) != len(upper_bound):
            raise ValueError("Lower and upper bounds must have the same length")

        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia = inertia
        self.b1 = b1
        self.b2 = b2
        self.fitness_func = fitness_func

        self.particles = [Particle(self.fitness_func, self.lower_bound, self.upper_bound) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')

        print("Initial particles")
        self.show_particles()

        for particle in self.particles:
            if particle.best_value < self.global_best_value:
                self.global_best_position = np.copy(particle.best_position)
                self.global_best_value = particle.best_value
    
    def optimize(self):
        """ Optimize the function """

        for iteration in range(self.num_iterations):
            
            # Update particles
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.inertia, self.b1, self.b2)
                particle.update_position(self.fitness_func, self.lower_bound, self.upper_bound)

            # Update global best
            for particle in self.particles:
                if particle.best_value < self.global_best_value:
                    self.global_best_position = np.copy(particle.best_position)
                    self.global_best_value = particle.best_value

            print(f"Iteration {iteration + 1}")
            print(f"Global Best Position: {[int(x) for x in self.global_best_position[:-1]] + [round(self.global_best_position[-1], 2)]}")
            print(f"Value: {self.global_best_value * -1}") 

            print("\n Table of particles")
            self.show_particles()

        return self.global_best_position, self.global_best_value * -1

    def show_particles(self):
        """ Show particles """
        
        table = []
        for id_, particle in enumerate(self.particles):
            table.append([
                id_, 
                [round(x, 2) for x in particle.position.tolist()],
                [round(x, 2) for x in particle.velocity.tolist()],
                [round(x, 2) for x in particle.best_position.tolist()],
                round(particle.best_value * -1, 6),
                round(particle.current_fitness * -1, 6)
            ])
        
        headers = ["Particle", "Position", "Velocity", "Best Position", "Best Value", "Current Fitness"]
        print(tabulate(table, headers = headers, tablefmt = "github"))       


# Load the dataset
def load_data():
    """ Load the dataset """
    try:
        df = pd.read_csv('data/heart_disease_dataset.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found. Please ensure the file is in the 'data/' directory.")

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# Define the fitness function
def fitness_func(params):
    """ Objective function """

    n_estimators, max_depth, min_samples_split, min_samples_leaf = params

    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = max(0.01, min(0.99, float(min_samples_leaf)))
    
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    return -accuracy

# Load the dataset
X_train_scaled, X_test_scaled, y_train, y_test = load_data()

# Define the lower and upper bounds
"""
n_estimators: [50, 200] number of trees in the forest
max_depth: [2, 50] maximum depth of the tree
min_samples_split: [2, 10] minimum number of samples required to split an internal node
min_samples_leaf: [0.01, 0.99] minimum number of samples required to be at a leaf node
"""
lower_bound = [50, 2, 2, 0.01]
upper_bound = [200, 50, 10, 0.99]  


pso = PSO(fitness_func, lower_bound, upper_bound)

best_params, best_value = pso.optimize()

print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")