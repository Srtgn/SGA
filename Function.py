'''

Created on 10.05.2020  @author: Saeed and Steffen Kastian

Simple Genetic Algorithm (Goldberg, 1989) code for truss size optimization 

'''
import numpy as np
import pandas as pd

class GA_truss(object):

    def __init__(self, truss_model):
        self.truss_model = truss_model
        self.Rho = 27.14 #Density KN/M**3
        self.a = 9.14  #  Length of Horizontal/Vertical truss's elements
        self.d = 12.925  # Length of Diagonal truss's elements
        self.equation_inputs = np.array([self.a, self.a, self.a, self.a, self.d, self.d, self.d, self.d, self.a, self.a])   #Todo: make it automatic

        self.Stress_Y = 172370 # yield stress KN/m**2
        self.P_factor = 1000 # a factor to magnify the penalty for elements with stress values higher than yield stress

    def calculate_fitness(self, generation, new_population):

        #Dimensions:
        sol_per_pop = new_population.shape[0]
        num_truss   = new_population.shape[1]

        # Initialize
        Result_stress = np.zeros((num_truss, sol_per_pop))
        fitness = np.zeros((num_truss,sol_per_pop))
        weight = np.zeros((num_truss,sol_per_pop))
        Displacement = np.zeros((sol_per_pop,1))

        # updating the area valued in the Geometry sheet
        for counter_pop, pop in enumerate(new_population):
            # saving the used elements' area for the analysed solution
            area = pop.reshape(pop.shape[0], 1)

            # Update the Geometric parameters of your model
            self.truss_model.Elements.geometric_properties = area

            # running the truss analysis to get the stress values
            self.truss_model.build()
            self.truss_model.run()

            # saving stress values of all solutions
            Result_stress[:, counter_pop] = np.atleast_1d(self.truss_model.Results.R[-1].element_stress[:])[:, 0]
            Displacement[counter_pop] = self.truss_model.Results.R[-1].U[5,1]

            # Calculate and save fitness
            fitness[:,counter_pop] = area[:,0] * self.equation_inputs *self.Rho
            weight [:,counter_pop]= fitness[:,counter_pop]

        # finding the index of the elements with stress higer than yield stress
        P_indices = np.where(abs(Result_stress) >= self.Stress_Y)
        
        # updating the fitness/weight value of the elements with stress higher than yeld stress
        fitness[P_indices] = (fitness[P_indices] + (self.P_factor * (abs(Result_stress[P_indices]) / self.Stress_Y)))

        Result_stress = Result_stress

        Weight = np.sum(weight, axis=0)
        B_Weight = np.min(Weight) * 100 # Converting KN to kg
        print('Best Weight', B_Weight)


        Fitness = np.sum(fitness, axis=0)
        Best_Fit = np.min(Fitness)
        Max_Disp = np.min(abs(Displacement))*100/2.54 # that's the only value in inch
        print('Max_Disp' , Max_Disp)

        B_index = np.where(Fitness == Best_Fit)
        B_stress = Result_stress[:,B_index[0][0]]
        B_population = new_population[B_index[0][0]]
        
        return Fitness, B_index, Best_Fit, B_Weight, B_stress, Result_stress, B_population, Max_Disp


def select_mating_pool(Population , Fitness  , num_parents ):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    Parent_Idx = sorted(range(len(Fitness)), key=lambda k: Fitness[k])[:num_parents]
    Parents = Population[Parent_Idx]
    return Parents
    
    
def crossover(parents, offspring_size):
    offspring1 = np.empty(offspring_size)
    offspring2 = np.empty(offspring_size)

    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]

        # The first offspring will have its first half of its genes taken from the first parent.
        offspring1[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The first offspring will have its second half of its genes taken from the second parent.
        offspring1[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
        # The second offspring will have its first half of its genes taken from the first parent.
        offspring2[k, 0:crossover_point] = parents[parent2_idx, 0:crossover_point]
        # The second offspring will have its second half of its genes taken from the second parent.
        offspring2[k, crossover_point:] = parents[parent1_idx, crossover_point:]
        offspring = np.concatenate((offspring1 , offspring2) , axis=0)
        
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = np.random.randint(0, mutations_counter - 1, 1)
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(.0006, 0.0018, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
            
    return offspring_crossover

def mutation_op(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = np.random.randint(0, mutations_counter - 1, 1)
#         gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(0.0003, 0.0006, 1)
#             offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] - random_value
            offspring_crossover[idx, gene_idx] = np.multiply(offspring_crossover[idx, gene_idx], 0.95)

            gene_idx = gene_idx + mutations_counter
            
    return offspring_crossover

def mutation_op2(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = np.random.randint(0, mutations_counter - 1, 1)
        for mutation_num in range(num_mutations):

            offspring_crossover[idx, gene_idx] = np.multiply(offspring_crossover[idx, gene_idx], 0.5)
            gene_idx = gene_idx + mutations_counter
            
    return offspring_crossover

def mutation_wise(offspring_crossover,Result_stress, num_mutations=1):
    
    for idx in range(offspring_crossover.shape[0]):
        gene_idx1 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),0)[0])          
        gene_idx2 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),1)[1])
        gene_idx3 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),2)[2])
        gene_idx4 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),3)[3])
        gene_idx5 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),4)[4])          
        gene_idx6 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),5)[5])
        gene_idx7 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),6)[6])
        gene_idx8 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),7)[7])
        gene_idx9 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),8)[8])          
        gene_idx10 = np.where(abs(Result_stress[:,idx]) == np.partition(abs(Result_stress[:,idx]),9)[9])


        offspring_crossover[idx, gene_idx1] = np.multiply(offspring_crossover[idx, gene_idx1], 0.8)
        offspring_crossover[idx, gene_idx2] = np.multiply(offspring_crossover[idx, gene_idx2], 0.85)
        offspring_crossover[idx, gene_idx3] = np.multiply(offspring_crossover[idx, gene_idx3], 0.9)
        offspring_crossover[idx, gene_idx4] = np.multiply(offspring_crossover[idx, gene_idx4], 0.95)
        offspring_crossover[idx, gene_idx5] = np.multiply(offspring_crossover[idx, gene_idx5], 1)
        offspring_crossover[idx, gene_idx6] = np.multiply(offspring_crossover[idx, gene_idx6],1)
        offspring_crossover[idx, gene_idx7] = np.multiply(offspring_crossover[idx, gene_idx7], 1)
        offspring_crossover[idx, gene_idx8] = np.multiply(offspring_crossover[idx, gene_idx8], 1)
        offspring_crossover[idx, gene_idx9] = np.multiply(offspring_crossover[idx, gene_idx9], 1)
        offspring_crossover[idx, gene_idx10] = np.multiply(offspring_crossover[idx, gene_idx10], 1)
            
    return offspring_crossover


def plot_nodes_numbers(nodeCords):
    x = [i[0] for i in nodeCords]
    y = [i[1] for i in nodeCords]
    size = 400
    offset = size/4000.
    plt.scatter(x, y, c='y', s=size, zorder=5)
    for i, location in enumerate(zip(x,y)):
        plt.annotate(i+1, (location[0]-offset, location[1]-offset), zorder=10)
        

def dfs_tabs(df_list, sheet_list, file_name):
    
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, header = True, index = False)   
    writer.save()