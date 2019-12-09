#Author: Matthew Osmond <mmosmond@ucdavis.edu>
#Description: Probability of rescue for different rates of wildtype decline

import numpy as np
import time
import csv
from scipy.stats import semicircular

######################################################################
##HELPER FUNCTIONS##
######################################################################

def open_output_files(N0, n, U, Es, mmax, mwt, nReps):
	"""
	This function opens the output files and returns file
	handles to each.
	"""
	sim_id = 'N%d_n%d_U%.5f_Es%.2f_mmax%.2f_mwt%.1f_nreps%d' %(N0, n, U, Es, mmax, mwt, nReps)
	data_dir = 'data'
	outfile_A = open("%s/dfe_platy_%s.csv" %(data_dir,sim_id),"w")
	return outfile_A

def write_data_to_output(fileHandles, data):
	"""
	This function writes data to files
	"""
	writer = csv.writer(fileHandles)
	writer.writerow(data)

def close_output_files(fileHandles):
	"""
	This function closes all output files.
	"""
	fileHandles.close()

def fitness(pop, mut, opt, B):
	"""
	This function calculates fitness
	"""
	phenos = np.dot(pop, mut) #sum mutations held by each individual to get phenotypes
	# dist = np.linalg.norm(phenos - opt, axis=1) #get phenotypic distances from optimum
	dist = np.sqrt(np.einsum('ij,ij->i', phenos - opt, phenos - opt)) #older python (maybe a bit faster)
	# dist = np.apply_along_axis(np.linalg.norm, 1, phenos - opt) #even older python (much slower)
	relW = np.exp(-0.5 * dist**2) #relative fitness: fraction of max offspring each individual is expected to produce
	W = relW*B #absolute fitness: expected number of offspring for each individual
	return W

def mitosis(pop, W):
	"""
	This function creates haploid offspring through asexual reproduction (ie mitosis)
	"""
	RW = np.random.poisson(W) #realized absolute fitness: number of gametes
	pop = np.repeat(pop, RW, axis=0) #make the gametes
	return pop

def mutation(pop, U, n, mut):
	"""
	This function creates mutations
	"""
	###############
	# max mutations a poisson number (U is expected number) - MUCH SLOWER!
	###############
	# nnewmuts = np.random.poisson(U, len(pop)) #poisson number of new mutations for each individual
	# oldmuts = np.sum(pop, axis=1) - 1 #number of mutations (not counting origin mutation) already in each individual (from ancestors)
	# totmuts = nnewmuts + oldmuts #total number of mutations in each individual
	# rtotmuts = np.clip(totmuts, 0, mutmax) #realized total number of mutations: allow each individual to have some max number of mutations
	# rnnewmuts = np.clip(rtotmuts - oldmuts, 0, None) #realized number of new mutations
	# newmutpop = np.repeat(np.transpose(np.identity(len(pop), dtype=int)), rnnewmuts, axis=1) #columns for new mutations in pop
	# pop = np.append(pop, newmutpop, axis=1) #append new columns to population matrix
	# newmuts = np.random.multivariate_normal(mean_mut, cov_mut, sum(rnnewmuts)) #phenotypic effect of new mutations
	# mut = np.append(mut, newmuts, axis=0) #append effect of new mutations to mutation matrix
	###############
	# max one new mutation per individual (U is a probability) - MUCH FASTER!
	###############
	rand3 = np.random.uniform(size = len(pop)) #random uniform number in [0,1] for each potential mutant
	nmuts = sum(rand3 < U) # mutate if random number is below mutation probability; returns number of new mutations
	whomuts = np.where(rand3 < U) #indices of mutants
	newmuts = semicircular.rvs(loc=0, scale=(4*l)**0.5, size=n*nmuts).reshape(nmuts,n) #phenotypic effect of new mutations
	pop = np.append(pop, np.transpose(np.identity(len(pop), dtype=int)[whomuts[0]]), axis=1) #add new loci and identify mutants
	mut = np.append(mut, newmuts, axis=0) #append effect of new mutations to mutation list
	###############
	return [pop, mut]

def remove_muts(remove_lost, pop, mut):
	"""
	This function removes lost mutations, if desired
	"""
	if remove_lost:
		keep = pop.any(axis=0)
		mut = mut[keep]
		pop = pop[:, keep]
	return [pop, mut]

######################################################################
##PARAMETERS##
######################################################################

#population parameters
N0 = 10**4 #initial number of wildtype (positive integer)
n = 4 #number of phenotype dimensions (positive integer)
Es = 10**(-2) #mean fitness effect of a random mutation (positive real)
l = 2 * Es / n #mutational variance (positive real)
U = 2*10**(-3) #mutation probability (positive real)
mmax = 0.5 #maximum malthusian growth rate (real)
B = np.exp(mmax) #max expected number of gametes produced (positive real)
mwts = [-0.1] #malthusian growth rate of wildtype (real) - list to loop over

#meta-parameters
rescue_N = 10**2 #number of individuals with positive growth rate needed to consider the population rescued (positive integer)
maxgen = 10**6 #max gens just in case no rescue or extinction (shouldnt be needed)
nReps = 10**5 #number of replicates (positive integer)
remove_lost = True #If true, remove mutations that are lost (0 for all individuals)

######################################################################
##SIMULATION##
######################################################################

def main():

	# loop over wildtype growth rates
	for mwt in mwts:

		# open output file
		fileHandles = open_output_files(N0, n, U, Es, mmax, mwt, nReps) 

		rep = 1
		while rep < nReps + 1:

			# initialize
			pop = np.array([[1]] * N0) #all start with same mutation at first locus
			mut = np.array([[0] * n]) #but let the mutation do nothing (ie put all phenotypes at origin)
			opt = np.append((2*(np.log(B)-mwt))**(0.5), [0]*(n-1)) #set optimum such that individuals at the origin have growth rate mwt
			W = fitness(pop, mut, opt, B) #fitness

			# start simulation
			gen = 0
			while gen < maxgen + 1:

				# end simulation if extinct      
				if len(pop) < 1: 
					print(mwt, rep, "Extinct")              
					break 

				# end simulation if rescued        
				if sum(W > 1) > rescue_N:			
					print(mwt, rep, "Rescued")
					genos, counts = np.unique(pop, axis=0, return_counts=True) #get unique genotypes and the number of individuals with each
					W = fitness(genos, mut, opt, B) #get fitness of each unique genotype
					rescues = W > 1 #focus on potential rescue genotypes
					ind = np.argmax(counts[rescues]) #choose most common as the rescue genotype
					rescue_growth = np.log(W[rescues][ind]) #growth rate of rescue genotype
					rescue_geno = genos[rescues][ind] #rescue genotype
					rescue_muts = np.sum(rescue_geno) - 1 #number of mutations in rescue genotype
					first_mut = mut[rescue_geno==1][1] #first mutation in rescue genotype
					dist = np.sqrt(np.einsum('i,i->', first_mut - opt, first_mut - opt)) #euclidean distance of mutation to optimum
					m_1 = np.log(B * np.exp(-0.5 * dist**2)) #growth rate of first mutation
					write_data_to_output(fileHandles, [rescue_growth, rescue_muts, m_1])                  
					break 

				# fitness
				W = fitness(pop, mut, opt, B)

				# mitosis
				pop = mitosis(pop, W)

				# mutation
				[pop, mut] = mutation(pop, U, n, mut)

				# remove lost mutations
				[pop, mut] = remove_muts(remove_lost, pop, mut)

				# go to next generation
				gen += 1

			# next replicate run
			rep += 1

	# cleanup
	close_output_files(fileHandles)

######################################################################
##RUNNING##
######################################################################    
	
# 	run (with timer)
start = time.time()
main()
end = time.time()
print(end-start)

