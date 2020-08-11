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


![Screenshot (221)](https://user-images.githubusercontent.com/51674837/89954730-ea432a80-dc31-11ea-9c5e-6ae0d1a95119.png)

![Screenshot (225)](https://user-images.githubusercontent.com/51674837/89954726-e8796700-dc31-11ea-8241-38ccaf2c864e.png)

