# SGA

##### Simple genetic algorithm (SGA) proposed by Goldberg in 1989:
This is a size optimization code, developed to be verified with the Takao YOKOTA (1989) 10-element truss. 
Please use the trusspy of the repository.

Its general procedure consist in the following steps:  
a. [Start] Generate random population of n chromosome (strings of 101010100110101, each of one correspond to a potential solution of the problem) 

b. [Fitness] Evaluate the fitness function f(x) for each chromosome in the population

c. [New population] Create the new population picking parents among the best individuals applying the (GA) operators:
    i. [Selection] Selection of two parents from a population according to their fitness (best fitness, more chance to be selected etc.) 
    ii. [Crossover] Generate children by mixing the parents properties with a crossover probability. If no crossover is applied the string is an exact copy of the parents.
    iii. [Mutation] Apply with a mutation probability changes to the children properties at each locus.  iv. [Accepting] Place the new strings in the population.  
    
d. [Replace] Use new generated population for a further run for the algorithm.  

e. [Test] If the end condition is satisfied, stop and give the best solution in the current population.

![Screenshot (217)](https://user-images.githubusercontent.com/51674837/89951796-5d49a280-dc2c-11ea-925d-407c2e903903.png)

![Screenshot (218)](https://user-images.githubusercontent.com/51674837/89951805-5fabfc80-dc2c-11ea-9b4c-70c49aaaf91b.png)
