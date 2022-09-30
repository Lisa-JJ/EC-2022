################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import glob, os
import random
import heapq

# multiprocessing
import concurrent.futures 
import multiprocessing
import concurrent.futures
from tkinter import E


# COPIED!!!
# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
# COPIED!!!

# experiment_name = 'EA2_dynamicCMrate_enemy1'
experiment_name = 'EA2_enemy8_check_run1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# For know using the neural network that is provided. NN: 1 hidden layer, consisting of 5 hidden notes.
hidden_notes = 10
selection_operator = 'elitism' # 'non_elitism'

# initializes environment with ai player using random controller, playing against static enemy
# COPIED!!!
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(hidden_notes),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini='yes')

env.state_to_log() # checks environment state
ini = time.time()  # sets time marker
# run_mode = 'train'
run_mode = 'test'
# COPIED!!!

# SAME AS IN CODE PROVIDED
def population_initialisation(lower_bound_weights, upper_bound_weights, population_size, chromosome_length):
    return np.random.uniform(lower_bound_weights, upper_bound_weights, (population_size, chromosome_length)) # use uniform distribution to sample

def population_evaluation(x):
    return np.array(list(map(lambda y: game_simulation(env,y), x)))

def game_simulation(env, x):
    fitness, player_life, enemy_life, game_runtime = env.play(pcont=x) # enemy taking random actions
    return fitness

# SAME AS IN CODE PROVIDED

def parent_selection(population_size, mating_pool_size, pop, fit_pop, mating_tournament_sample_size):
    # # uniform parent selection
    # return random.randint(0,population_size-1)

    # tournament parent selection (fav of prof), creating mating pool
    mating_pool, fit_mating_pool = tournament_selection(pop, fit_pop, population_size, mating_tournament_sample_size, mating_pool_size)
    return mating_pool, fit_mating_pool

def tournament_selection(pop, fit_pop, population_size, tournament_sample_size, selection_pool_size):
    pool = []
    fit_pool = []
    current_member = 0
    while (current_member<=(selection_pool_size-1)):
        # without replacement
        potential_parents = random.sample(range(population_size), tournament_sample_size)
        winner = 1000
        fit_winner = -1000
        for p in potential_parents:
            if fit_pop[p] > fit_winner:
                fit_winner = fit_pop[p]
                winner = pop[p]
        pool.append(winner)
        fit_pool.append(fit_winner)   
        current_member += 1
    return pool, fit_pool 

def crossover(pop, population_size, recombination_probability, chromosome_length, mating_pool_size, fit_pop, mating_tournament_sample_size, recombination_scale):
    # single arithmetic recombination
    alpha = recombination_probability
    parents, fit_parents = parent_selection(population_size, mating_pool_size, pop, fit_pop, mating_tournament_sample_size)
    offspring = single_arithmetic_recombination(parents, population_size, chromosome_length, recombination_probability)
    # offspring = blend_recombination(parents, population_size, chromosome_length, recombination_probability, recombination_scale)
    return offspring, parents, fit_parents

def blend_recombination(parents, population_size, chromosome_length, recombination_probability, recombination_scale):
    offspring = []
    for i in range(int(population_size*1.5)):
        parent1 = parents[random.randint(0,len(parents)-1)]
        parent2 = parents[random.randint(0,len(parents)-1)]
        if np.random.uniform(0,1) <= recombination_probability:
            gamma = ((1-(2*recombination_scale))*np.random.uniform(0,1))-recombination_scale
            # child1 = [((1-gamma)*parent1[i]) + (gamma*parent2[i]) for i in range(chromosome_length)]
            # child2 = [((1-gamma)*parent2[i]) + (gamma*parent1[i]) for i in range(chromosome_length)]
            child1 = parent1
            child2 = parent2
            for i in range(0,chromosome_length):
                child1[i] = ((1-gamma)*parent1[i]) + (gamma*parent2[i])
                child2[i] = ((1-gamma)*parent2[i]) + (gamma*parent1[i])
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)
    return offspring

def whole_arithmetic_recombination(parents, population_size, chromosome_length, recombination_probability):
    offspring = []
    # generates double the amount of offspring than population size
    for i in range(0,population_size-1):
        parent1 = parents[random.randint(0,len(parents)-1)]
        parent2 = parents[random.randint(0,len(parents)-1)]
        if np.random.uniform(0,1) <= recombination_probability:
            child1 = recombination_scale*parent1 + (1-recombination_scale)*parent2
            child2 = recombination_scale*parent2 + (1-recombination_scale)*parent1
            offspring.append(child1)
            offspring.append(child2)
        else: 
            offspring.append(parent1)
            offspring.append(parent2)
    return offspring

def single_arithmetic_recombination(parents, population_size, chromosome_length, recombination_probability):
    offspring = []
    # generates double the amount of offspring than population size
    for i in range(0,population_size-1):
        parent1 = parents[random.randint(0,len(parents)-1)]
        parent2 = parents[random.randint(0,len(parents)-1)]
        if np.random.uniform(0,1) <= recombination_probability:
            k = random.randint(0,chromosome_length-1)
            # child1 = recombination_scale*parent1 + (1-recombination_scale)*parent2
            # child2 = recombination_scale*parent2 + (1-recombination_scale)*parent1
            child1 = parent1
            child2 = parent2
            child1[k] = recombination_scale*parent1[k] + (1-recombination_scale)*parent2[k]
            child2[k] = recombination_scale*parent2[k] + (1-recombination_scale)*parent1[k]
            offspring.append(child1)
            offspring.append(child2)
        else: 
            offspring.append(parent1)
            offspring.append(parent2)
    return offspring

def mutation(offspring, mutation_probability, mutation_std):
    # nonuniform selection
    for i in range(0,len(offspring)):
        if np.random.uniform(0,1) <= mutation_probability:
            for gene in range(0,len(offspring[i])):
                # perform mutation
                if np.random.uniform(0,1) <= mutation_probability:
                    offspring[i][gene] = curtailing(offspring[i][gene]+np.random.normal(0,mutation_std))
    return offspring

def curtailing(gene):
    if gene < lower_bound_weights:
        return lower_bound_weights
    elif gene > upper_bound_weights:
        return lower_bound_weights
    else:
        return gene

def survivor_selection(pop, fit_pop, tournament_sample_size, selection_operator):
    if selection_operator == 'elitism':
        pop_new, fit_pop_new = round_robin_tournament(pop, fit_pop, population_size, tournament_sample_size)
    else:
        pop_new, fit_pop_new = elitism(pop, fit_pop, population_size)
    return pop_new, fit_pop_new

def round_robin_tournament(pop, fit_pop, population_size, tournament_sample_size):
    pop_wins = [0]*len(pop)
    for p in range(0,len(pop)-1):
        competitor = random.sample([i for i in range(population_size) if i != p], tournament_sample_size)
        for c in competitor:
            if fit_pop[p] > fit_pop[c]:
                pop_wins[p] += 1
    index_pop_new = heapq.nlargest(population_size, range(len(pop_wins)), pop_wins.__getitem__)
    pop_new = [pop[i] for i in index_pop_new]
    fit_pop_new = [fit_pop[i] for i in index_pop_new]
    return pop_new, fit_pop_new

def elitism(pop, fit_pop, population_size):
    pool = sorted(range(len(fit_pop)), key = lambda sub: fit_pop[sub])[-population_size:]
    # pool = heapq.nlargest(population_size, zip(fit_pop, pop))
    pop_new = [pop[p] for p in pool]
    fit_pop_new = [fit_pop[p] for p in pool]
    return pop_new, fit_pop_new





#last_best = 0
# Set necessary parameters
observation_space = env.get_num_sensors()
bias = 1
action_space = 5
chromosome_length = (observation_space + bias)*hidden_notes + (hidden_notes + bias)*action_space

# TO BE TUNED 
generations = 20                    # lecturer mentioned >= 30
population_size = 100                # change accordingly

lower_bound_weights = -1            # change accordingly
upper_bound_weights = 1             # change accordingly
recombination_scale = 0.5
# recombination_probability = 0.2     # change accordingly (before 0.5)
# mutation_probability = 0.8          # change accordingly (before 0.1)
recombination_probability = 1     # change accordingly (before 0.5)
mutation_probability = 0.2          # change accordingly (before 0.2)
mutation_std = 0.01                    # change accordingly
mating_pool_size = 50               # goes in dependency with population_size
mating_tournament_sample_size = 5   # the higher the higher the selection pressure
survivor_tournament_sample_size = 5 # 10 typical value

# COPIED!!!
# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    # env.update_parameter('speed','normal')
    population_evaluation([bsol])

    sys.exit(0)

# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = population_initialisation(lower_bound_weights, upper_bound_weights, population_size, chromosome_length)
    fit_pop = population_evaluation(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0                       # initial generation
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()
# COPIED!!!
# np.savetxt(experiment_name+'/best_init.txt',pop[best])








# Evolution
last_sol = fit_pop[best]
notimproved = 0
for i in range(1,generations):
    # parent selection in crossover
    offspring, parents, fit_parents = crossover(pop, population_size, recombination_probability, chromosome_length, mating_pool_size, fit_pop, mating_tournament_sample_size, recombination_scale)
    offspring = mutation(offspring, mutation_probability, mutation_std)
    fit_offspring = population_evaluation(offspring)
    
    # COPIED !!!
    # combine original population and offspring to gain new population for survivor selection
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)
    # best = np.argmax(fit_pop) #best solution in generation
    # fit_pop[best] = float(population_evaluation(np.array([pop[best]]))[0]) # repeats best eval, for stability issues
    # best_sol = fit_pop[best]
    # COPIED !!!
    pop, fit_pop = survivor_selection(pop, fit_pop, survivor_tournament_sample_size, selection_operator)

    # recombination_probability += 0.02     # change accordingly (before 0.5)
    # mutation_probability -= 0.02

    # COPIED !!!
    # searching new areas
    # if best_sol <= last_sol:
    #     notimproved += 1
    # else:
    #     last_sol = best_sol
    #     notimproved = 0

    # if notimproved >= 15:

    #     file_aux  = open(experiment_name+'/results.txt','a')
    #     file_aux.write('\ndoomsday')
    #     file_aux.close()

    #     pop, fit_pop = doomsday(pop,fit_pop)
    #     notimproved = 0
    # COPIED !!!

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)

    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    # np.savetxt(experiment_name+'/best'+str(i)+'.txt',pop[best])
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()


# COPIED!!!
fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state
# COPIED!!!
