using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV


 
# Chapter 7.4.2 - Random Time Effects ~~~~~~~~~~~~~~~~~~~~

#= 
From the book:
The model shown in 7_4_1 treats time as a fixed-effects factor; for every
occasion, an independent effect is estimated. To asses the temporal variability,
we cannot simply take these fixed-effects estimates and calculate
their variance. By doing so, we would ignore the fact that these values
are estimates that have an unknown associated error. Thus, we would 
assume that there is no sampling variance, and this can hardly ever be true. 
However, when treating time as a random-effects factor, we can separate sampling
(ie. variance within years) from process variance (ie variance between years),
exactly as we did in the state-space models in chapter 5.
=#




# Simulate Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

n_occ = 20                      # seasons 
marked = repeat([30], n_occ-1)  # individuals marked each season
mean_phi = 0.65
var_phi = 1
p = repeat([0.4], n_occ-1)

# Logit and inv logit functions
logit(p) = log(p/(1-p))
inv_logit(x) = (1+tanh(x/2))/2

# create distribution to draw survival from and produce draws
surv_dist = Normal(qlogis(mean_phi), var_phi^0.5)
logit_phi = rand(surv_dist, n_occ-1)
phi = inv_logit.(logit_phi)

# generate matrices, n_ind x n_occ
PHI = repeat(transpose(phi), inner=(sum(marked),1))
P = repeat(transpose(p), inner=(sum(marked),1))

function sim_cjs(PHI, P, marked) 
    n_occ = size(PHI)[2] + 1
    CH = zeros(Int64, sum(marked), n_occ)
    mark_occ = [repeat([i],marked[i]) for i=1:length(marked)]
    mark_occ = collect(Iterators.flatten(mark_occ)) # flatten into vector

    # Loop through individuals and time steps to draw from survival and resighting probs
    for i in 1:sum(marked)
        CH[i, mark_occ[i]] = 1 # mark the first capture/marking occasion
           if mark_occ[i] == n_occ; continue 
           end
        for t in mark_occ[i]+1:n_occ
            surv = rand(Binomial(1, PHI[i,t-1]), 1)
                if surv == [0]; break # if individual dies, move on
                end
            rp = rand(Binomial(1, P[i,t-1]), 1)
                if rp == [1]; CH[i,t] = 1 # if individual survived, and was resighted, insert a 1 in 
                end
        end
    end
    return CH
end

CH = sim_cjs(PHI, P, marked)



# helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Probability of individual being alive and not captured.
function prob_uncaptured(n_ind, n_occ, p, phi) 
    z = Matrix{Real}(undef, n_ind, n_occ)
    for i in 1:n_ind
        z[i, n_occ] = 1.0;
        for t in 1:n_occ - 1
            t_curr = n_occ - t;
            t_next = t_curr + 1;
            z[i, t_curr] = (1 - phi[i, t_curr]) + phi[i, t_curr] * (1 - p[i, t_next - 1]) * z[i, t_next];
        end
    end
    return z        
end

function first_capture(y_i)
    for k in 1:length(y_i)
        if y_i[k] == 1
            return k
        end
    end
end

function last_capture(y_i) 
    for k_rev in 0:(length(y_i) - 1)
      k = length(y_i) - k_rev;
      if y_i[k] == 1
        return k
      end
    end
end



# Model Specification ~~~~~~~~~~~~~~~~~~~

@model cjs_cc_marg(y) = begin
    n_ind,n_occ = size(y)

    # Create empty matrices
    phi = Matrix{Real}(undef, n_ind, n_occ-1)
    p   = Matrix{Real}(undef, n_ind, n_occ-1)
    first = Vector{Int}(undef, n_ind)
    last  = Vector{Int}(undef, n_ind)
    
    
    # calculate first and last capture occasions
    for i in 1:n_ind
        first[i] = first_capture(y[i,:])
        last[i] = last_capture(y[i,:])
    end
    
    # ~~~~ Priors ~~~~~~~~~~~~~~~~~~~~~~~
    mean_phi ~ Uniform(0,1)
    mean_p   ~ Uniform(0,1)
    sigma   ~ Uniform(0,10)
    epsilon ~ filldist(Normal(0, sigma), n_occ-1)
    
    # Constraints on phi and p / transformed parameters

    #= 
    The prior choices for μ and for σ² need some thought. Because
    μ is the mean survival on the logit scale, a noninformative prior on the
    logit scale would be a normal distribution with wide variance. Yet, this 
    prior will not be noninformative on the probability scale. In the code 
    below we provide two options: first, a normal distribution with wide 
    variance for μ, and second, a uniform distribution for logit⁻¹(μ), which 
    is noninformative on the probability scale but informative on the logit scale.
    =#
    
    mu = logit(mean_phi)
   
    for i in 1:n_ind
        for t in 1:first[i]-1
            phi[i,t] = 0
            p[i,t] = 0
        end
        
        for t in first[i]:n_occ - 1
            phi[i,t] = inv_logit(mu + epsilon[t])
            p[i,t] = mean_p     
        end
    end
    
    z = prob_uncaptured(n_ind, n_occ, p, phi)
    
    # ~~~~ Likelihood ~~~~~
    for i in 1:n_ind
        
        if first[i] > 0 
            for t in (first[i]+1):last[i]
                1 ~ Bernoulli(phi[i,t-1])
                y[i,t] ~ Bernoulli(p[i, t-1])
            end
        end
    1 ~ Bernoulli(z[i,last[i]])
    end
end


# Sample ~~~~~~~~~~~~~~~~~~~
y = CH
model = cjs_cc_marg(y)
chain = sample(model, NUTS(100, 0.65), 2500)


plot(chain)
