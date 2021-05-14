using Pkg
using Distributed
using DelimitedFiles
using Distances
using Profile    
using Random 
rng = MersenneTwister(1234)
using DelimitedFiles
using Statistics
using LinearAlgebra

include("./DCATools_KAI/basic_bmDCA.jl")
include("./DCATools_KAI/basic.jl")

function Monte_Carlo_adaptive(L::Int64, n::Int64, A::Array{Int64,1}, J::Array{Float64,2}, h::Array{Float64,1})
	n_accepted = 0
	for l=1:n
		i = rand(1:L)	
		(accepted, A) = Metropolis_Hastings(i, A, J, h)
		n_accepted += accepted
	end
	return (n_accepted,A) 
end

function read_param(q::Int64, L::Int64, fname::String)	
	h = zeros(L*q) 
	J = zeros(L*q, L*q) 
	parameters = readdlm(fname)
	@show (N0,N1) = size(parameters) # 	
	for n in 1:N0
		if(parameters[n,1]=="J")	
			i,j,a,b,v = parameters[n, 2]+1,parameters[n, 3]+1,parameters[n, 4]+1, parameters[n, 5]+1, parameters[n,6] 	
			J[(i-1)*q+a, (j-1)*q+b] = v	
			J[(j-1)*q+b, (i-1)*q+a] = v	
		end	
		if(parameters[n,1]=="h")	
			i,a,v = parameters[n, 2]+1, parameters[n, 3]+1, parameters[n, 4]
			h[(i-1)*q+a] = v
		end	
	end
	return (J, h) 
end

########## main #############
#q::Int64 = 21
q=21;q = Int64(q)
L=70
fname_para = "/data/shimagaki/sparse-BM-Analysis/sp76/parameters/Parameters_tmp_7975_PF76_sparse.dat"
(J, h) = read_param(q, L, fname_para)

Data = zeros(Int64, 2^13,L)
Data_old = zeros(Int64, 2^13,L)

#--------  MCMC parameters --------#
T_eq = 1e4
T_aut = 50 
n_branch = 13
nMC = L 
#----------------------------------#

#
#Maybe Monte_Carlo_sweep is too many MC steps.

#---------- Equilibration ---------#
A = rand(0:(q-1), L)	

for m=1:T_eq
	global A	
	(n_accepted, A) = Monte_Carlo_adaptive(L,L, A, J, h)# it is setted nMC=L
end
Data_old[1,:] = copy(A)
#----------------------------------#


leaf_id = 1
for n in 1:n_branch
	global Data_old, Data	
	
	leaf_id = 2^(n-1)
	for id in 1:leaf_id	
		#------- Sample1. ------#	
		A = Data_old[id,:] 
		for t=1:T_aut
			(n_accepted, A) = Monte_Carlo_adaptive(L,nMC, A, J, h)
		end	
		Data[2*id-1,:] = copy(A)
		
		#------- Sample2. ------#	
		A = Data_old[id,:] 
		for t=1:T_aut
			(n_accepted, A) = Monte_Carlo_adaptive(L,nMC, A, J, h)
		end	
		Data[2*id,:] = copy(A) 
	end
	Data_old = copy(Data)	
	println("Done: ID:", n)
end



fname_out = "statistics_correlated_Tau-"*string(T_aut)*"_Nbranch-"*string(n_branch)*"_nMC-"*string(nMC)*".dat" 
fout = open(fname_out, "w")
for n in 1:2^n_branch
	for i in 1:L
		print(fout, Data[n, i], " ")
	end
	println(fout)
end
close(fout)
"""


for epoch=1:10000
	global J,h,X_after_transition,k_max,M_msa,q,L, lambda_h, lambda_J, reg_h, reg_J, c2_msa, f1_msa
	(f1,f2, X_after_transition) = pCDk(X_after_transition, k_max, M_msa, q, L, J, h)
	(J, h, dh_sum, dJ_sum, cc, cslope, froc) = gradient_ascent(lambda_h, lambda_J, reg_h, reg_J, f1_msa, f1, f2, c2_msa, J, h)
	println(epoch, " ", dh_sum, " ", dJ_sum, " ", cc, " ", cslope, " ", froc)	
	
	if(epoch%500==0)
		output_paramters(epoch,L,q,h,J)
		output_statistics(epoch,L,q,5000,h,J)
	end

end
output_paramters(L,q)
"""
	
