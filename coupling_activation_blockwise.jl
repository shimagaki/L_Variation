#Pkg.add("Pkg"); import Pkg; 
using Pkg
using Distributed
using Distances
using StatsBase 
using Profile    
using Printf
using Plots
using Random
using Distributions
rng = MersenneTwister(1234)
using DelimitedFiles
using Statistics
using LinearAlgebra

include("/Users/kaishimgakki/Documents/Info-Tech/Programming/Tools_for_DCA/basic.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/Tools_for_DCA/basic_Hopfield.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/Tools_for_DCA/basic_analysis.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/Tools_for_DCA/basic_bmDCA.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/Tools_for_DCA/basic_rbmDCA.jl");
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/Tools_for_DCA/basic_generative.jl");


function get_f1_f2_persistent(L::Int64, q::Int64, t_weight::Int64,  X_persistent::Array{Int64, 2}, E_old_vec::Array{Float64,1}, h::Array{Float64, 1}, J::Array{Float64, 2}, f1_msa::Array{Float64, 1}, c_vec::Array{Float64, 1})
	f1=zeros(q*L)
	f2=zeros(q*L, q*L)
	
	n_sample = size(X_persistent,1)	
	ones_L = ones(Int64, L);
	myscale = 1.0/n_sample
    ones_L_float = myscale*ones(L);		
    ones_LL_float = myscale*ones(L,L);	


	E1_ave = 0.0; E2_ave = 0.0
	for m=1:n_sample
		A = copy(X_persistent[m,:])
		E_old = E_old_vec[m]
        for t=1:t_weight
			(n_accepted, A, E_old) = Monte_Carlo_sweep(E_old, q, L, A, J, h)
		end
        E_old_vec[m] = E_old
		X_persistent[m, :] = copy(A)
		f1[km.(1:L,A+ones_L, q)] += ones_L_float	
		f2[km.(1:L,A+ones_L, q), km.(1:L,A+ones_L, q)] += ones_LL_float	
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

	return (f1, f2, X_persistent, E_old_vec, E1_ave, E2_ave, mc, mslope, cc, cslope) 
end

function get_E(A::Array{Int64, 1}, J::Array{Float64, 2}, h::Array{Float64, 1})
	e_i = 0.0
	ones_L=ones(Int64, L)
	# The diagonal elements of J, J_ii should be zero-matricies. 
	e_i = - 0.5*sum( J[km.(1:L, A+ones_L, q), km.(1:L, A+ones_L, q)] )	
	e_i += -sum(h[km.(1:L, A+ones_L, q)])
	return e_i 
end

function get_f1_f2_with_pseudocount(q, L, f1_msa, f2_msa, alpha)
	scale_qq = 1.0/(q*q)	
	scale_q = 1.0/(q)	
	f1_output = (1-alpha)*copy(f1_msa) + alpha*scale_q *ones(q*L) 
	f2_output = (1-alpha)*copy(f2_msa) + alpha*scale_qq*ones(q*L, q*L) 
	return (f1_output, f2_output)
end

# epoch=1; largest, second largest. 
# Is the second largest J_ij can be the largest at the next epoch? ---> Test it. 
#suppose the f1_msa and f2_msa are already weighted by the alpha an
function get_coupling_J_likelihood_Block(alpha,  q, L, f1_msa, f2_msa, f2_model)
	scale_qq = 1.0/(q*q)	
	scale_q = 1.0/(q)	
	i_max=1;j_max=1;
	delta_l_block_max = -100;
	
	J_block_tot = zeros(q*L, q*L)	
	delta_l_block_tot = zeros(L, L)	
	MI_ele_tot = zeros(L, L)	
	for i in 1:L
		#f_d_i = (1-alpha)*f1_msa[km.(i,1:q,q)] + alpha*scale_q*ones(q)
		f_d_i = f1_msa[km.(i,1:q,q)]
		for j in (i+1):L
			#f_d_j = (1-alpha)*f1_msa[km.(i,1:q,q)] + alpha*scale_q*ones(q)
			f_d_j = f1_msa[km.(j,1:q,q)]
			
			#f_d = (1-alpha)*f2_msa[km.(i,1:q,q), km.(j,1:q,q)] + alpha*scale_qq*ones(q,q)
			f_d = f2_msa[km.(i,1:q,q), km.(j,1:q,q)]
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
	# Should use only the elements that are i<j	
	return (i_max, j_max, delta_l_block_tot, J_block_tot, MI_ele_tot) 
end

########### main #############
q=4;P=0
fname_in = "../MSA/PF14/MSA_artificial_q4_PF14.txt";
X_msa = readdlm(fname_in, Int)
@show M_msa,L = size(X_msa); 
W_msa = ones(M_msa) 

println("Done read the traing data.")
@time (Meff,L, f1_msa, f2_msa, c2_msa) = f1_f2_c2(X_msa, W_msa, q)
println("Done compute the basic statistics.");

fname_cotact_PF14 = "/Users/kaishimgakki/Documents/Info-Tech/Programming/artificial_Protein/data/PF00014_PF00014_g0_correlation_mindist_wDCA_fmt.txt"
#contact_true = read_dist(fname_cotact_PF14, q, L, 1,2,4);
contact_true = read_contact(fname_cotact_PF14, q, L, 1,2,4, 6);

alpha = 1e-3 #pseudocount alpha = 1e-30
(f1_msa, f2_msa) = get_f1_f2_with_pseudocount(q, L, f1_msa, f2_msa, alpha) 
c2_msa = f2_msa - f1_msa*f1_msa' 
for i in 1:L
	f2_msa[km.(i,1:q,q),km.(i,1:q,q)] = zeros(q,q)
	c2_msa[km.(i,1:q,q),km.(i,1:q,q)] = zeros(q,q)
end
c_vec_msa = vec(c2_msa);

h = log.(f1_msa + 0.1 / Meff * ones(q*L))
#(h, J) = init_h_J(f1_msa, q, L, Meff)
J = zeros(q*L, q*L)
J_filter = zeros(Int64, L, L)
println("Done set the model paramters.");

fname_statistics_out = "series_of_statistics.txt"
fout_statistics = open(fname_statistics_out, "w")
fname_record_out = "added_record.txt"
fout_add_record = open(fname_record_out, "w")

n_TP = 0.0; n_FP = 0.0
t_weight = 5; 
t_eq = 1000
#n_sample = 10_000
epoch_max = 2700 # almost maximum.
interval_output = 300
n_coupling_added = 0
BIC_accum = 0.0
X_persistent = copy(X_msa);
n_sample = size(X_msa,1)
E_old_vec = zeros(n_sample)
for n in 1:n_sample
    E_old_vec[n] = E_i(q,L,1,rand(0:(q-1),L),J,h)
end
epoch_max=100
epoch = 1;

epoch_max=100
for epoch in 1:epoch_max
	global X_persistent, J, J_filter, h, f1_msa, f2_msa, c2_msa, c_vec_msa, n_coupling_added, BIC_accum, E_old_vec, contact_true
    (f1_samples, f2_samples, X_persistent, E_old_vec, E1_ave, E2_ave, mc, mslope, cc, cslope) = get_f1_f2_persistent(L, q, t_weight, X_persistent, E_old_vec, h, J, f1_msa, c_vec_msa);
    (i_max, j_max, delta_l_block_tot, J_block_tot, MI_ele_tot) = get_coupling_J_likelihood_Block(alpha, q, L, f1_msa, f2_msa, f2_samples);
    
    # --- Adding the couplings ----#
    J[km.(i_max, 1:q, q), km.(j_max, 1:q, q)] += copy(J_block_tot[km.(i_max, 1:q, q), km.(j_max, 1:q, q)])	
    # To use J_ji elements, J_ij matrix shuld be used after transporse.
    J[km.(j_max, 1:q, q), km.(i_max, 1:q, q)] += copy(J_block_tot[km.(j_max, 1:q, q), km.(i_max, 1:q, q)]')	

    # The definition of the BIC is the same as the article of Silvio and Federico. 
    BIC_local = 2*Meff*delta_l_block_tot[i_max, j_max]
    if(J_filter[i_max, j_max]==0)
        n_coupling_added += 1
        BIC_local += - q*q*log(Meff)  
        if(contact_true[i_max, j_max]==1)
            n_TP += 1.0
        end
    end
    J_filter[i_max, j_max] += 1; J_filter[j_max, i_max] += 1
    BIC_accum += BIC_local	
    
    @printf "J %d %d num_Jadd: %d d_l: %.3e BIC_local: %.3e BIC_accum: %.3e n_J_tot: %d mc: %.3e mslope: %.3e cc: %.3e cslope: %.3e heatC: %.3e ppv: %.3e \n" i_max j_max J_filter[i_max, j_max] delta_l_block_tot[i_max, j_max] BIC_local/Meff BIC_accum/Meff n_coupling_added mc mslope cc cslope E2_ave-E1_ave^2 n_TP/n_coupling_added 
    #BIC_mat[i_max, j_max] = BIC_local
end
