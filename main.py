'''

Created on 10.05.2020  @author: Saeed Rastegarian and Steffen Kastian

Simple Genetic Algorithm (Goldberg, 1989) code for truss size optimization 

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import trusspy from folder
import sys
sys.path.append(r'../../')
import trusspy as tp

from Function import*

# Input parameters
truss_input_file = "truss1.xlsx"
sol_per_pop = 8
num_parents_mating = 4
num_weights = 10

# Defining the population size.
pop_size = (sol_per_pop, num_weights)  

# Creating the initial population.
new_population = np.random.uniform(low=0.006, high=0.0085,
                                   size=pop_size)  

M = tp.Model(truss_input_file, logfile=False, log=0, print_term=False)
M.Settings.log=0
ga_truss = GA_truss(M)

# Generic Algorithm
best_outputs_fit = [15000]
best_outputs_W = [2000]

num_generations = 20000

file_name = 'first'
data = []
Dataframe = pd.DataFrame()

for generation in range(num_generations):
    
    print("Generation : ", generation)
    
    Fitness, B_index, Best_Fit, Best_Weight, B_stress, Result_stress, Best_population, Max_Disp = ga_truss.calculate_fitness(generation = generation, new_population= new_population)
    print('B_population', Best_population)

    output_dic = ({'Generation': generation,
        '1': Best_population[0], '2': Best_population[1],
        '3': Best_population[2], '4': Best_population[3],
        '5': Best_population[4], '6': Best_population[5],
        '7': Best_population[6], '8': Best_population[7],
        '9': Best_population[8], '10': Best_population[9],
        'weight': [Best_Weight],
        '01': B_stress[0], '02': B_stress[1],
        '03': B_stress[2], '04': B_stress[3],
        '05': B_stress[4], '06': B_stress[5],
        '07': B_stress[6], '08': B_stress[7],
        '09': B_stress[8], '010': B_stress[9]})
    
    df = pd.DataFrame(data = output_dic)

    Dataframe = Dataframe.append(df)

    best_outputs_fit.append((Best_Fit))
    best_outputs_W.append((Best_Weight))
    
    if Best_Weight > 1000:

        if Best_Fit > 1000:
        
            print('Designing the structure')
            parents = select_mating_pool(new_population, Fitness, num_parents_mating)

            offspring_crossover = crossover(parents, offspring_size=(num_parents_mating, num_weights))         

            offspring_mutation = mutation(offspring_crossover, num_mutations=2)

            new_population = offspring_mutation
            
        else:
            
                
            if (abs(best_outputs_W[generation-1] - best_outputs_W[generation])) >  (best_outputs_W[generation-1] * 0.0001):
                print('Weight optimization')   
    
                parents = select_mating_pool(new_population, Fitness, num_parents_mating)     
                offspring_crossover = crossover(parents, offspring_size=(num_parents_mating, num_weights))
                offspring_mutation_wise = mutation_wise(offspring_crossover,Result_stress, num_mutations=1)
                new_population = offspring_mutation_wise
    
            else:
                   
                print('Generation {} is the best solution' .format(generation))
                break
#         
    else:
         
        if Best_Fit < 1000:
             
            if (abs(best_outputs_W[generation-1] - best_outputs_W[generation])) >=  (best_outputs_W[generation-1] * 0.0001):
                print('Weight optimization qweqweqwe')   
      
                parents = select_mating_pool(new_population, Fitness, num_parents_mating)     
                offspring_crossover = crossover(parents, offspring_size=(num_parents_mating, num_weights))
                offspring_mutation_wise = mutation_wise(offspring_crossover,Result_stress, num_mutations=1)
                new_population = offspring_mutation_wise
              
            else:
                  
                print('Generation {} is the best solution' .format(generation))
                break
         
#         else:
#               
#             print('Designing the structure')
#             parents = select_mating_pool(new_population, Fitness, num_parents_mating)
#             offspring_crossover = crossover(parents, offspring_size=(num_parents_mating, num_weights))         
#             offspring_mutation = mutation(offspring_crossover, num_mutations=2)
#             new_population = offspring_mutation
            

Dataframe.to_excel(".\output.xlsx")

plt.plot(best_outputs_W) 
plt.show()

