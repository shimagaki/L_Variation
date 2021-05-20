#Pkg.add("Pkg"); import Pkg; 
using Pkg
using Distributed
using Distances
using StatsBase 
using Profile    
using Random
using Distributions
rng = MersenneTwister(1234)
using DelimitedFiles
using Statistics
using LinearAlgebra
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia_temp/tools/basic_rbmDCA.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia_temp/tools/basic.jl")

function get_f1_f2(L::Int64, q::Int64, n_sample::Int64, t_weight::Int64, t_eq::Int64, A::Array{Int64, 1},  h::Array{Float64, 1}, J::Array{Float64, 2})
	X = zeros(Int64, n_sample, L)	
	f1=zeros(q*L)
	f2=zeros(q*L, q*L)
	for m=1:t_eq
		(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
	end	
	ones_L = ones(Int64, L);	
	ones_LL = ones(Int64, L,L);	
	myscale = 1.0/n_sample
	E1_ave = 0.0; E2_ave = 0.0
	for m=1:n_sample
		for t=1:t_weight
			(n_accepted, A) = Monte_Carlo_sweep(L, A, J, h)
		end
		f1(km.(1:L,A+ones_L, q)) += myscale*ones_L	
		f2(km.(1:L,A+ones_L, q), km.(1:L,A+ones_L, q)) += myscale*ones_LL	
		E_temp = get_E(A, J, h)
		E1_ave += E_temp
		E2_ave += E_temp*E_temp
	end
	for i in 1:L
		f2[km.(i,1:q,q),km.(i,1:q,q)] = zeros(q,q)
	end
	E1_ave /= n_sample; E2_ave /= n_sample

	return (f1, f2, E1_ave, E2_ave, A) 
end

function get_E(A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	e_i = 0.0
	ones_L=ones(Int64, L)
	# The diagonal elements of J, J_ii should be zero-matricies. 
	e_i = - 0.5*sum( J[km.(1:L, A+ones_L, q), km.(1:L, A+ones_L, q)] )	
	e_i += -sum(h[km.(1:L, A+ones_L, q)])
	return e_i 
end

function get_J_opt_Likelihood_Variation(alpha,  q, L, f1_msa, f2_msa, f2_model)
	scale = 1.0/(q*q)	
	scale1 = 1.0/(q)	
	J_list = []
	J_list = zeros(map(Int64,q*q*L*(L-1)/2), 9)	
	n =1 
	for i in 1:L
		for j in (i+1):L
			delta_l_of_J_block=0.0	
			MI_ele= 0.0	
			for a in 1:q
				for b in 1:q
					f_d = (1-alpha)*f2_msa[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					f_m =(1-alpha)*f2_model[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					delta_l_of_J_block += f_d * log(f_d / f_m )	
					
					f_d_a =  (1-alpha)*f1_msa[(i-1)*q+a]+alpha*scale1	
					f_d_b =  (1-alpha)*f1_msa[(j-1)*q+b]+alpha*scale1	
					
					MI_ele += f_d * log(f_d / (f_d_a*f_d_b) )	
				end
			end
		
		
			for a in 1:q
				for b in 1:q
					f_d = (1-alpha)*f2_msa[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					f_m =(1-alpha)*f2_model[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					
					delta_l_of_J_elem = f_d * log(f_d/f_m) +(1-f_d) * log( (1-f_d) / (1-f_m) )	
					J_elem = log( (f_d * (1-f_m)) / (f_m * (1-f_d)) ) 	
					J_block = log( f_d / f_m ) 	
					#The likelihood varidation should be discussed only on the couplings that is not introduced. 
					#take into account only zeros couplings. 
				
					J_list[n,1],J_list[n,2] = i,j 
					J_list[n,3],J_list[n,4] = a,b 
					
					J_list[n,5] = J_elem
					J_list[n,6] = delta_l_of_J_elem
					J_list[n,7] = J_block
					J_list[n,8] = delta_l_of_J_block
					J_list[n,9] = MI_ele 
					n += 1	
				end
			end	
				
		end
	end
	
	#J_list = J_list'
	@show size(J_list) # should be (q*q*L*(L-1)/2, 5) 
	#id_list = sortperm(J_list[:,1])
	#return J_list[id_list, :]	
	return J_list
end

# epoch=1; largest, second largest. 
# Is the second largest J_ij can be the largest at the next epoch? ---> Test it. 
#
function get_coupling_J_likelihood_Block(alpha,  q, L, f1_msa, f2_msa, f2_model)
	scale_qq = 1.0/(q*q)	
	scale_q = 1.0/(q)	
	n =1 
	@show "size should be q x q"
	@show size(ones(q)*ones(q)'')
	i_max=1;j_max=1;
	delta_l_block_max = -100;
	for i in 1:L
		f_d_i = (1-alpha)*f1_msa[km.(i,1:q,q)] + alpha*scale_q*ones(q)
		for j in (i+1):L
			if(J_filter[i,j]==0)	
				f_d_j = (1-alpha)*f1_msa[km.(i,1:q,q)] + alpha*scale_q*ones(q)
				
				f_d = (1-alpha)*f2_msa[km.(i,1:q,q), km.(j,1:q,q)] + alpha*scale_qq*ones(q,q)
				f_m = (1-alpha)*f2_model[km.(i,1:q,q), km.(j,1:q,q)] + alpha*scale_qq*ones(q,q)
		
				# --- Block-wise coupling additions --- #
				delta_l_block = sum( f_d .* log.( f_d ./ f_m ) )	
				J_block = log.(f_d ./ f_m)
				
				# We can also keep only the coupling associate with i_max and j_max
				if(delta_l_block_max < delta_l_block)
					delta_l_block_max = delta_J_block
					i_max = i; j_max = j
				end

				# --- Element-wise coupling additions --- #
				#delta_l_elem = f_d .* log.( f_d ./ f_m ) + (ones(q,q) - f_d) .* log.( (ones(q,q) - f_d) ./ (ones(q,q) - f_m))
				#J_elem = log.(f_d .* (ones(q,q) - f_m) ./ (f_m .* (ones(q,q) - f_d) ) )

				MI_ele = sum( f_d .* log.( f_d ./ (f_d_i * f_d_j') ) )	
			end
		end
	end
	return (i_max, j_max, delta_l_block, J_block, MI_ele) 
end




function output_paramters_adding_couplings(t::Int64, L::Int64, q::Int64, P::Int64, h::Array{Float64, 1}, xi::Array{Float64,2}, J::Array{Float64,2})
	fname_out = "./parameters-t"*string(t)*"_add_couplings.txt"
	fout = open(fname_out, "w")
	(n_max, n_ele) = size(J)
	J_out = zeros(q*L, q*L)	
	for n in 1:n_max
		i, j, a, b, = map(Int64,J[n, 1]),map(Int64,J[n, 2]) ,map(Int64,J[n, 3]),map(Int64,J[n, 4])
		J_elem, delta_J_elem, J_block, delta_J_block, MI_ele = J[n, 6], J[n, 6], J[n, 7], J[n, 8], J[n, 9] 
		println(fout, "J ", i-1, " ", j-1, " ", a-1, " ", b-1, " ", J_elem, " ", delta_J_elem, " ", J_block, " ", delta_J_block, " ", MI_ele)	
	end
	
	for i=1:L
		for mu=1:P 
			for a=1:q
				println(fout, "xi ", i-1, " ", mu-1, " ", a-1, " ", xi[(i-1)*P+mu, a])
			end
		end
	end
	
	for i=1:L
		for a=1:q
			println(fout, "h ", i-1, " ", a-1, " ", h[(i-1)*q+a])
		end
	end
	close(fout)
end

########### main #############
q=4;P=0

X_msa = readdlm("/Users/kaishimgakki/Documents/Info-Tech/Programming/artificial_Protein/note/MSA_artificial_q4_PF14.txt", Int)
M_msa,L = size(X_msa); W_msa = ones(M_msa) 

println("here is the result of the msa.")
@time (Meff,L, f1_msa, f2_msa, c2_msa) = f1_f2_c2(X_msa, W_msa, q)
k_max = 2 
lambda_h,lambda_xi = 0.1, 1.0 
reg_h, reg_xi = 1e-3, 1e-3 #reg_xi=1e-1 works well -> NO! Hiddens go zeros!  

h = log.(f1_msa+0.0001*ones(size(f1_msa)))  

fname_error_out = "error_log.txt"
fout_error = open(fname_error_out, "w")

t_weight = 30; 
#can be t_eq = 1, after some epoch. 
t_eq = 1000
n_sample = 30_000
for epoch=0:3_000
	global X_msa, xi,h,X_after_transition,k_max,M_msa,q,L, lambda_h, lambda_xi, reg_h, reg_xi, f1_msa, f2_msa, c2_msa
	
	alpha = 1e-30 #pseudocount
	(f1_samples, f2_samples, E1_ave, E2_ave, A) = get_f1_f2(L, q, n_sample, t_weight, t_eq, A, h, J,)
	
	J_opt = get_coupling_J_likelihood_Block(alpha, q, L, f1_msa, f2_msa, f2_samples)
	output_paramters_adding_couplings(epoch,L,q,P,  h, xi, J_opt)
	#end
end

close(fout_error)
