from numpy import array, dot, mean
from numpy.linalg import pinv
from math import floor

from sklearn import preprocessing
from random import random, sample, choice

from tqdm import tqdm
from sys import exit

def calc_w(X, Y):
    """
    Calculate weights

    Parameters
    ----------
    A : numpy array
        matrix resulted from independent vector 
    y : numpy array
        vector of dependent values

    Returns
    -------
    numpy array
        numpy array with calculated weights
    """
    return dot((pinv((dot(X.T, X)))), (dot(X.T, Y)))

def linear_regression(inputs, outputs):
    """
    Get the best expected outcome.
    """
    X, Y = array(inputs), array(outputs)
    w = calc_w(X,Y)
    Y_pred = dot(X, w)
    Y_mean = mean(Y)
    
    # metrics
    SST = array([(y - Y_mean) ** 2 for y in Y]).sum()                       # sum of all deviations squared
    SSR = array([(y - y_pred) ** 2 for y, y_pred in zip(Y, Y_pred)]).sum()  # sum of squared residuals
    COD = (1 - (SSR / SST))                                                 # coefficient of determination
    average_error = (SSR / len(Y))
    
    return {'COD': COD, 'weights': w, 'error': average_error}


def check_termination_condition(best_individual, COD_threshold, max_reached):
    """
    Check if the current_best_individual is better of equal to the expected.
    """
    if ((best_individual['COD'] >= COD_threshold)
            or (max_reached)):
        return True
    else:
        return False


def create_individual(individual_size):
    """
    Create an individual.
    """
    return [random() for i in range(individual_size)]


def create_population(individual_size, population_size):
    """
    Create an initial population.
    """
    return [create_individual(individual_size) for i in range(population_size)]


def get_fitness(individual, inputs, outputs):
    """
    Calculate the fitness of an individual.
    Return the Coefficient of Determination, average error and weight.
    We use the error to get the best individual.
    """
    predicted_outputs = dot(array(inputs), array(individual))
    output_mean = mean(outputs)
    
    # metrics
    SST = array([(y - output_mean) ** 2 for y in outputs]).sum()
    SSR = array([(y - y_pred) ** 2 for y, y_pred in zip(outputs, predicted_outputs)]).sum()
    COD = (1 - (SSR / SST))
    average_error = (SSR / len(outputs))
    
    return {'COD': COD, 'error': average_error, 'weights': individual}


def evaluate_population(population, inputs, outputs, 
                        selection_size, best_individuals_stash,
                        test_input, test_output,
                        history, generation_count):
    """
    Evaluate a population of individuals and return the best among them.
    """
    fitness_list = [get_fitness(individual, inputs, outputs)
                    for individual in tqdm(population)]
    
    error_list = sorted(fitness_list, key=lambda i: i['error'])
    best_individuals = error_list[: selection_size]
    best_individuals_stash.append(best_individuals[0]['weights'])
    
    print('Error (Train): ', best_individuals[0]['error'],
          'COD (Train): ', best_individuals[0]['COD'])
    
    current_best_individual_test = get_fitness(best_individuals_stash[-1], test_input, test_output)
    
    print('Error (Test): ', current_best_individual_test['error'],
          'COD (Test): ', current_best_individual_test['COD'])
    
    history[generation_count] = [best_individuals[0]['error'],  
                                 current_best_individual_test['error'], 
                                 best_individuals[0]['weights']]
    
    return best_individuals, history


def crossover(parent_1, parent_2, individual_size):
    """
    Return offspring given two parents.
    Unlike real scenarios, genes in the chromosomes aren't necessarily linked.
    """
    child = {}
    
    loci = [i for i in range(0, individual_size)]
    loci_1 = sample(loci, floor(0.5*(individual_size)))
    loci_2 = [i for i in loci if i not in loci_1]
    
    chromosome_1 = [[i, parent_1['weights'][i]] for i in loci_1]
    chromosome_2 = [[i, parent_2['weights'][i]] for i in loci_2]
    
    child.update({key: value for (key, value) in chromosome_1})
    child.update({key: value for (key, value) in chromosome_2})
    
    return [child[i] for i in loci]


def mutate(individual, individual_size, probability_of_gene_mutating):
    """
    Mutate an individual.
    The gene transform decides whether we'll add or deduct a random value.
    """
    loci = [i for i in range(0, individual_size)]
    no_of_genes_mutated = floor(probability_of_gene_mutating*individual_size)
    loci_to_mutate = sample(loci, no_of_genes_mutated)
    
    for locus in loci_to_mutate:
        gene_transform = choice([-1, 1])
        change = gene_transform*random()
        individual[locus] = individual[locus] + change
        
    return individual


def get_new_generation(selected_individuals, 
                       population_size, 
                       individual_size, 
                       probability_of_individual_mutating,
                       probability_of_gene_mutating):
    """
    Given selected individuals, create a new population by mating them.
    Here we also apply variation operations like mutation and crossover.
    """
    parent_pairs = [sample(selected_individuals, 2)
                    for i in range(population_size)]
    
    offspring = [crossover(pair[0], pair[1], individual_size) for pair in parent_pairs]
    offspring_indices = [i for i in range(population_size)]
    offspring_to_mutate = sample(
        offspring_indices,
        floor(probability_of_individual_mutating*population_size)
    )
    
    mutated_offspring = [[i, mutate(offspring[i], individual_size, probability_of_gene_mutating)]
                         for i in offspring_to_mutate]
    
    for child in mutated_offspring:
        offspring[child[0]] = child[1]
        
    return offspring

if __name__ == "__main__":
    pass