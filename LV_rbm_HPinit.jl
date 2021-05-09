#Pkg.add("Pkg"); import Pkg; 
using Pkg
using Distributed
using Distances
using StatsBase 
using Profile    
using Random
using Distributions
using DelimitedFiles
using Statistics
using LinearAlgebra
using Plots

rng = MersenneTwister(1234)
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia/dca_tools/basic_rbmDCA.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia/dca_tools/basic.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia/dca_tools/basic_analysis.jl")
include("/Users/kaishimgakki/Documents/Info-Tech/Programming/dca_tools_Julia/dca_tools/basic_Hopfield.jl")

function output_statistics_temp(L::Int64, P::Int64, n_sample::Int64, n_weight::Int64,  xi::Array{Float64, 2}, h::Array{Float64, 1}, f1_msa::Array{Float64, 1}, f2_msa::Array{Float64, 2}, c2_msa::Array{Float64, 2})
	A_model = rand(0:20, L)	
	H_model=zeros(P)
	for m=1:1000
		H_model = sampling_hidden(P,L,A_model,xi)
		A_model = sampling_visible(q,L,P, H_model, h, xi)
	end	
	X_output = zeros(Int64, n_sample, L)	
	
	for m=1:n_sample
		for t=1:n_weight
			H_model = sampling_hidden(P,L,A_model,xi)
			A_model = sampling_visible(q,L,P, H_model, h, xi)
		end
		
		for i=1:L
			X_output[m,i] = A_model[i]	
		end

	
	end
	(M_eff, L, f1_out, f2_out, c2_out) = f1_f2_c2(X_output, ones(n_sample),q)	
	return (f1_out, f2_out)
end


function get_J_opt_Likelihood_Variation(alpha, th, q, L, f1_msa, f2_msa, f2_model)
	scale = 1.0/q^2	
	scale1 = 1.0/q	
	J_list_Int = zeros(Int64, map(Int64, q*q*L*(L-1)/2), 4)	
	J_list_Float = zeros(map(Int64, q*q*L*(L-1)/2), 5)	
	n =1 
	for i in 1:L
		for j in (i+1):L
			delta_l_of_J_block=0.0	
			MI_ele= 0.0	
			for a in 1:q
				for b in 1:q
					f_d = (1-alpha)*f2_msa[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					f_m =(1-alpha)*f2_model[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					if(f_d>th && f_m>th)	
						delta_l_of_J_block += f_d * log(f_d / f_m )	
						f_d_a_b =  (1-alpha)*f1_msa[(i-1)*q+a]*f1_msa[(j-1)*q+b]+alpha*scale	
						MI_ele += f_d * log(f_d / (f_d_a_b) )	
					end	
				end
			end
		
		
			for a in 1:q
				for b in 1:q
					f_d = (1-alpha)*f2_msa[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					f_m =(1-alpha)*f2_model[(i-1)*q+a, (j-1)*q+b]+alpha*scale 
					
					J_elem=0.0; delta_l_of_J_elem=0.0; J_block=0.0	
					if(f_d>th && f_m>th)	
						delta_l_of_J_elem = f_d * log(f_d/f_m) +(1-f_d) * log( (1-f_d) / (1-f_m) )	
						J_elem = log( (f_d * (1-f_m)) / (f_m * (1-f_d)) ) 	
						J_block = log( f_d / f_m ) 	
					end	
					J_list_Int[n,1],J_list_Int[n,2] = i,j 
					J_list_Int[n,3],J_list_Int[n,4] = a,b 
					
					J_list_Float[n,1] = J_elem
					J_list_Float[n,2] = delta_l_of_J_elem
					J_list_Float[n,3] = J_block
					J_list_Float[n,4] = delta_l_of_J_block
					J_list_Float[n,5] = MI_ele 
					n += 1	
				end
			end	
		end
	end
	return (J_list_Int, J_list_Float) 
end

function output_paramters_adding_couplings(fname_out::String, L::Int64, q::Int64, P::Int64, h::Array{Float64, 1}, xi::Array{Float64,2}, J_opt_Int::Array{Int64,2}, J_opt_Float::Array{Float64,2})
	fout = open(fname_out, "w")
	n_max = size(J_opt_Int,1)
	J_out = zeros(q*L, q*L)	
	for n in 1:n_max
		i, j, a, b, = J_opt_Int[n, 1], J_opt_Int[n, 2] , J_opt_Int[n, 3], J_opt_Int[n, 4]
		J_elem, delta_J_elem, J_block, delta_J_block, MI_ele = J_opt_Float[n, 1], J_opt_Float[n, 2], J_opt_Float[n, 3], J_opt_Float[n, 4], J_opt_Float[n, 5] 
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

function convert_xi(q::Int64, L::Int64, p::Int64, xi)
	xi_convert = zeros(L*P, q)

	for i in 1:L
		for m in 1:P
			for a in 1:q
				xi_convert[(i-1)*P+m, a] = xi[m,km(i,a,q)]
			end
		end
	end
	return xi_convert
end

########### main #############
q=4
P=3
N_max = 1000
lambda_h,lambda_xi =0.02, 0.02 
reg_h, reg_xi = 1e-3, 1e-3 #reg_xi=1e-1 works well -> NO! Hiddens go zeros!  

fname_seq = "/Users/kaishimgakki/Documents/Info-Tech/Programming/artificial_Protein/note/MSA_artificial_q4_PF14.txt"  
X_msa = readdlm(fname_seq, Int)[1:N_max,:]
@show M_msa,L = size(X_msa)
### Parameters for HP model ###
P_p=P+1; P_m=P+1
file_key="PF14_q"*string(q)*"_M"*string(N_max)
fname_fig="HP_eigenvalues.png"
th_MSA=1.1
pseudo_para = 0.01
###############################


W_msa = ones(M_msa) 
println("here is the result of the msa.")
@time (Meff,L, f1_msa, f2_msa, c2_msa) = f1_f2_c2(X_msa, W_msa, q)
k_max = 2 
h = log.(f1_msa+0.0001*ones(size(f1_msa)))  
(Xi_p, Xi_m, p1, p2) = main_HP(q, L, fname_seq, file_key, fname_fig, P_p, P_m, th_MSA, pseudo_para);
xi = copy( Xi_p[2:(P+1), :])

xi = convert_xi(q,L,P, xi)
xi_vec_temp = 0;

(f1_temp,f2_temp, psi_data_temp, psi_model_temp, X_after_transition) = pCDk_rbm(q, L, P, 
									    M_msa, k_max, 
									    h, xi, 
									    X_msa, X_msa)
fname_error_out = "error_log.txt"
fout_error = open(fname_error_out, "w")

n_sample = 10_000
n_weight = 30

n_divi = 2
d_n_batch = Int( floor(M_msa / float(n_divi)) )
for epoch=1:8_000
	global X_msa, xi,h,X_after_transition,k_max,M_msa,q,L, lambda_h, lambda_xi, reg_h, reg_xi, f1_msa, f2_msa, c2_msa
	
	for n_b in 1:n_divi	
		id_set = collect(( (n_b-1) * d_n_batch+1):(n_b*d_n_batch) )	

		
		(f1, f2, psi_data, psi_model, X_after_transition) = pCDk_rbm_minibatch(q, L, P,
									 M_msa, k_max,
									 id_set,
									 h, xi, 
									 X_msa, X_after_transition)
		
		
		(xi,h, 
		sum_dh, sum_dxi, 
		sum_dh2, sum_dxi2, 
		cc,cslope,froc) = gradient_ascent(q, L, P, 
						lambda_h, lambda_xi, 
						reg_h, reg_xi, 
						f1_msa, f1, 
						f2_msa, f2, 
						psi_data, psi_model, 
						h, xi)
		
		
		println(epoch, " ",sum_dh, " ", sum_dxi, " ",sum_dh2, " ", sum_dxi2, " ",  cc, " ", cslope, " ", froc)	
		
		println(fout_error,  epoch, " ",sum_dh, " ", sum_dxi, " ",sum_dh2, " ", sum_dxi2, " ",  cc, " ", cslope, " ", froc)	

	end

	if(epoch%2000==0)
		fname_out = "/data/shimagaki/Adding_couplings/PF76_init_HP/peq"*string(P)*"/parameters-t"*string(epoch)*"_add_couplings.txt"
		#n_sample = 5000
		#th = 1e-5	
		th = 0	
		alpha = 1e-1 #pseudocount	
		(f1_samples, f2_samples) = output_statistics_temp(L,P,n_sample,n_weight, xi,h, f1_msa, f2_msa, c2_msa)
		
		(J_opt_Int, J_opt_Float) = get_J_opt_Likelihood_Variation(alpha, th, q, L, f1_msa, f2_msa, f2_samples)
		
		output_paramters_adding_couplings(fname_out,L,q,P,  h, xi, J_opt_Int, J_opt_Float)
	end
end

close(fout_error)


