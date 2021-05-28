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
function get_coupling_J_likelihood_Elem(alpha,  q, L, f1_msa, f2_msa, f2_model)
	scale_qq = 1.0/(q*q)	
	scale_q = 1.0/(q)	
	
	J_elem_tot = zeros(q*L, q*L)	
	delta_l_elem_tot = zeros(L*q, L*q)	
	
	qq = q*q
	ones_qq = ones(qq) 
	J_graph_vec = zeros(Int(L*(L-1)/2), 5)	
	n_J_graph = 1

	for i in 1:L
		f_d_i = f1_msa[km.(i,1:q,q)]
		for j in (i+1):L
			f_d_j = f1_msa[km.(j,1:q,q)]
			
			f_d = f2_msa[km.(i,1:q,q), km.(j,1:q,q)]
			f_m = (1-alpha)*f2_model[km.(i,1:q,q), km.(j,1:q,q)] + alpha*scale_qq*ones(q,q)
		
			# --- Block-wise coupling additions --- #
			#delta_l_block = sum( f_d .* log.( f_d ./ f_m ) )	
			#J_block = log.(f_d ./ f_m)
			
			# --- Element-wise coupling additions --- #
			delta_l_elem = f_d .* log.( f_d ./ f_m ) + (ones(q,q) - f_d) .* log.( (ones(q,q) - f_d) ./ (ones(q,q) - f_m))
			J_elem = log.(f_d .* (ones(q,q) - f_m) ./ (f_m .* (ones(q,q) - f_d) ) )
		
			#exclude those are already included in.
			#delta_l_elem = delta_l_elem .* (ones(q,q)-J_filter[km.(i,1:q,q), km.(j,1:q,q)])	
			
			# --- Maybe I can keep the largest. ---#
			# I should add the coupling elements that are not included in the model.
			# We can also keep only the coupling associate with i_max and j_max
			ids = argmax(delta_l_elem)
			a,b = ids[1], ids[2]
			if(delta_l_elem[a,b] > 0) # if(delta_l_elem[a,b]>0 is True) => it is not yet added.
				J_graph_vec[n_J_graph, 1] = i
				J_graph_vec[n_J_graph, 2] = j
				J_graph_vec[n_J_graph, 3] = a 	
				J_graph_vec[n_J_graph, 4] = b 
				J_graph_vec[n_J_graph, 5] = delta_l_elem[a, b]	
				n_J_graph += 1	
			end	
			
			
			#MI_ele = sum( f_d .* log.( f_d ./ (f_d_i * f_d_j') ) )	
			#J_block_tot[km.(i,1:q, q),km.(j,1:q, q)] = copy(J_block) 
			#delta_l_block_tot[i,j] = delta_l_block
			#MI_ele_tot[i,j] = MI_ele
			
			J_elem_tot[km(i,a,q),km.(j,b,q)] = J_elem[a,b]
			delta_l_elem_tot[km(i,a,q),km.(j,b,q)] = delta_l_elem[a,b]
		end
	end
	
	#J_graph_vec = sortslices(J_graph_vec, dims=1, by=x=->x[5], rev=true)# This should be the descending order.
	J_graph_vec = copy(J_graph_vec[sortperm(J_graph_vec[:, 5], rev=true), :] )	
	# Should use only the elements that are i<j	
	return (J_graph_vec, delta_l_elem_tot, J_elem_tot) 
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
J_filter = zeros(Int64, q*L, q*L)
println("Done set the model paramters.");

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
n_add_coupling_max = 1

epoch_max=100
for epoch in 1:epoch_max
	global X_persistent, J, J_filter, h, f1_msa, f2_msa, c2_msa, c_vec_msa, n_coupling_added, BIC_accum, E_old_vec, contact_true
    (f1_samples, f2_samples, X_persistent, E_old_vec, E1_ave, E2_ave, mc, mslope, cc, cslope) = get_f1_f2_persistent(L, q, t_weight, X_persistent, E_old_vec, h, J, f1_msa, c_vec_msa);
    #(i_max, j_max, delta_l_block_tot, J_block_tot, MI_ele_tot) = get_coupling_J_likelihood_Block(alpha, q, L, f1_msa, f2_msa, f2_samples);
    (J_graph_vec, delta_l_elem_tot, J_elem_tot) = get_coupling_J_likelihood_Elem(alpha, q, L, f1_msa, f2_msa, f2_samples) 
    
	# --- Adding the couplings ----#
	BIC_local = 0.0
	for n in 1:n_add_coupling_max	
		i,j,a,b = Int(J_graph_vec[n,1]), Int(J_graph_vec[n,2]), Int(J_graph_vec[n,3]), Int(J_graph_vec[n,4])
		J[km(i,a,q), km(j,b,q)] += J_elem_tot[km(i,a,q), km(j,b,q)]
		J[km(j,b,q), km(i,a,q)] += J_elem_tot[km(i,a,q), km(j,b,q)]
		
		BIC_local = 2 * Meff*delta_l_elem_tot[km(i,a,q), km(j,b,q)] 
		if(J_filter[km(i,a,q), km(j,b,q)]==0)
			n_coupling_added += 1
			BIC_local +=  - log(Meff)  
            if(contact_true[i, j]==1)
                n_TP += 1.0
            end
		end
		J_filter[km(i,a,q), km(j,b,q)] += 1 
		J_filter[km(j,b,q), km(i,a,q)] += 1
		
		BIC_accum += BIC_local     
    

    
        #@printf "J %d %d num_Jadd: %d d_l: %.3e BIC_local: %.3e BIC_accum: %.3e n_J_tot: %d mc: %.3e mslope: %.3e cc: %.3e cslope: %.3e heatC: %.3e ppv: %.3e \n" i_max j_max J_filter[i_max, j_max] delta_l_block_tot[i_max, j_max] BIC_local/Meff BIC_accum/Meff n_coupling_added mc mslope cc cslope E2_ave-E1_ave^2 n_TP/n_coupling_added 
        @printf "J %d %d %d %d Jadded: %.3e num_Jadd: %d d_l: %.3e BIC_local: %.3e BIC_accum: %.3e n_J_tot: %d mc: %.3e mslope: %.3e cc: %.3e cslope: %.3e heatC: %.3e ppv: %.3e \n" i j a b J_elem_tot[km(i,a,q), km(j,b,q)] J_filter[km(i,a,q), km(j,b,q)] delta_l_elem_tot[km(i,a,q), km(j,b,q)] BIC_local/Meff BIC_accum/Meff n_coupling_added mc mslope cc cslope E2_ave-E1_ave^2 n_TP/n_coupling_added 
    end
    #if(epoch%interval_output == 0)
	#	output_paramters_bm(epoch, L, q, h, J, J_filter)
	#end
end
