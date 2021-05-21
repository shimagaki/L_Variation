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
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia_temp/tools/basic_analysis.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia_temp/tools/basic_bmDCA.jl")

function get_f1_f2(L::Int64, q::Int64, n_sample::Int64, t_weight::Int64, t_eq::Int64, A::Array{Int64, 1},  h::Array{Float64, 1}, J::Array{Float64, 2}, f1_msa::Array{Float64, 1}, c_vec::Array{Float64, 1})
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
		f1[km.(1:L,A+ones_L, q)] += myscale*ones_L	
		f2[km.(1:L,A+ones_L, q), km.(1:L,A+ones_L, q)] += myscale*ones_LL	
		E_temp = get_E(A, J, h)
		E1_ave += E_temp
		E2_ave += E_temp*E_temp
	end
	E1_ave /= n_sample; E2_ave /= n_sample
	
	c2 = f2-f1*f1'	
	for i in 1:L
		f2[km.(i,1:q,q),km.(i,1:q,q)] = zeros(q,q)
		c2[km.(i,1:q,q),km.(i,1:q,q)] = zeros(q,q)
	end
	c_vec = vec(c2)
	
	mc = Statistics.cor(f1_msa, f1)	
	mslope = linreg(f1_msa, f1)[2]	
	
	cc = Statistics.cor(c_vec_msa, c_vec)	
	cslope = linreg(c_vec_msa, c_vec)[2]	

	return (f1, f2, A, E1_ave, E2_ave, mc, mslope, cc, cslope) 
end

function get_E(A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	e_i = 0.0
	ones_L=ones(Int64, L)
	# The diagonal elements of J, J_ii should be zero-matricies. 
	e_i = - 0.5*sum( J[km.(1:L, A+ones_L, q), km.(1:L, A+ones_L, q)] )	
	e_i += -sum(h[km.(1:L, A+ones_L, q)])
	return e_i 
end

# epoch=1; largest, second largest. 
# Is the second largest J_ij can be the largest at the next epoch? ---> Test it. 
#
function get_coupling_J_likelihood_Block(alpha,  q, L, f1_msa, f2_msa, f2_model, J_filter)
	scale_qq = 1.0/(q*q)	
	scale_q = 1.0/(q)	
	@show "size should be q x q"
	@show size( ones(q)*ones(q)' )
	i_max=1;j_max=1;
	delta_l_block_max = -100;
	
	J_block_tot = zeros(q*L, q*L)	
	delta_l_block_tot = zeros(L, L)	
	MI_ele_tot = zeros(L, L)	
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
					delta_l_block_max = delta_l_block
					i_max = i; j_max = j
				end

				# --- Element-wise coupling additions --- #
				#delta_l_elem = f_d .* log.( f_d ./ f_m ) + (ones(q,q) - f_d) .* log.( (ones(q,q) - f_d) ./ (ones(q,q) - f_m))
				#J_elem = log.(f_d .* (ones(q,q) - f_m) ./ (f_m .* (ones(q,q) - f_d) ) )

				MI_ele = sum( f_d .* log.( f_d ./ (f_d_i * f_d_j') ) )	
				
				J_block_tot[km.(i,1:q, q),km.(j,1:q, q)] = copy(J_block) 
				delta_l_block_tot[i,j] = delta_l_block
				MI_ele_tot[i,j] = MI_ele
			end
		end
	end
	return (i_max, j_max, delta_l_block_tot, J_block_tot, MI_ele_tot) 
end

function output_paramters_adding_couplings(t::Int64, L::Int64, q::Int64, P::Int64, h::Array{Float64, 1}, J::Array{Float64,2})
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
X_msa = readdlm("/Users/kaishimgakki/Documents/Info-Tech/Programming/artificial_Protein/note/MSA_artificial_q4_PF14.txt", Int)[1:5000, :]

@show M_msa,L = size(X_msa); 
W_msa = ones(M_msa) 

println("Done read the traing data.")
@time (Meff,L, f1_msa, f2_msa, c2_msa) = f1_f2_c2(X_msa, W_msa, q)
println("Done compute the basic statistics.")
c_vec_msa = vec(c2_msa)

lambda_h,lambda_xi = 0.1, 1.0 
reg_h, reg_xi = 1e-3, 1e-3 #reg_xi=1e-1 works well -> NO! Hiddens go zeros!  

(h, J) = init_h_J(f1_msa, q, L, Meff)
J_filter = zeros(Int64, L, L)
println("Done set the model paramters.")

fname_error_out = "error_log.txt"
fout_error = open(fname_error_out, "w")

t_weight = 30; 
t_eq = 1000
n_sample = 10_000
epoch_max = 10

A = rand(0:(q-1), L)
#likelihood_traject = zeros(epoch_max)
#pearson_traject = zeros(epoch_max)
#slope_traject = zeros(epoch_max)
#heatcap_traject = zeros(epoch_max)
BIC_traject = zeros(epoch_max)


#Note each time compare is not the empirical frequencies, but the model frequencies that are added the coplings. 
for epoch=1:epoch_max
	global A, J, J_filter, h, f1_msa, f2_msa, c2_msa, c_vec_msa
	
	alpha = 1e-30 #pseudocount alpha = 1e-30
	
	(f1_samples, f2_samples, A, E1_ave, E2_ave, mc, mslope, cc, cslope) = get_f1_f2(L, q, n_sample, t_weight, t_eq, A, h, J, f1_msa, c_vec_msa)
	
	(i_max, j_max, delta_l_block_tot, J_block_tot, MI_ele_tot) = get_coupling_J_likelihood_Block(alpha, q, L, f1_msa, f2_msa, f2_samples, J_filter)
	J_filter[i_max, j_max] = 1; J_filter[j_max, i_max] = 1
	
	# The definition of the BIC is the same as the article of Silvio and Federico. 
	BIC_local = 2 * Meff*delta_l_block_tot[i_max, j_max] - log(Meff)  
	if(epoch==1)
		BIC_traject[epoch] = BIC_local
	end
	if(epoch>1)
		BIC_traject[epoch] = BIC_traject[epoch-1] + BIC_local
	end
	
	@show mc, mslope, cc, cslope, E2_ave-E1_ave^2, BIC_local 
	@show i_max, j_max, delta_l_block_tot[i_max, j_max], MI_ele_tot[i_max, j_max] 
	
	#output_paramters_adding_couplings(epoch, L, q, P, h, J)
	#end
end

close(fout_error)
