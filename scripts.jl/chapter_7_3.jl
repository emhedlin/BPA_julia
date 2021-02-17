using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV

threads 4
 
# Chapter 7.3 - Models with Constant Parameters ~~~~~~~~~~


# Simulate Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

n_occ = 6                       # seasons 
marked = repeat([50], n_occ-1)  # individuals marked each season
phi = repeat([0.65], n_occ-1)   # survival probability from one t-1 to t 
p = repeat([0.4], n_occ-1)      # resighting probability

PHI = repeat([phi[1]], sum(marked), n_occ-1)
P = repeat([p[1]], sum(marked), n_occ-1)


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
    
    # data dimensions
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
    
        # ~~~~ Priors ~~~~
    mean_phi ~ Uniform(0,1)
    mean_p   ~ Uniform(0,1)
    
    # Constraints on phi and p
    for i in 1:n_ind
        for t in 1:first[i]-1
            phi[i,t] = 0
            p[i,t] = 0
        end
        
        for t in first[i]:n_occ - 1
            phi[i,t] = mean_phi # this is where you add linear models
            p[i,t] = mean_p     # this is where you add linear models
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
chain = sample(cjs_cc_marg(y),  NUTS(100, 0.65), 100, save_state = false)

y

plot(chain)





