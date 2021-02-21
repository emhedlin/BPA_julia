using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV
using StatsFuns: logistic

#= 
single season occupancy, varying parameters

WIP
=#

# Spatial and Temporal replication
R, T = 200, 3

# Process Parameters
ψ, p = 0.5, 0.3

# observations
z = zeros(R) # empty array
z = rand!(Binomial(1, ψ), z) #fill empty array with latent state
z_mat = [z z z]





y = zeros(R,T)
# Observation process
for i in 1:R 
    for t in 1:T 
        y[i,t] = (z_mat[i,t] * rand(Binomial(1, p), 1))[1]
    end
end

sum(z_mat)
sum(y)

n_sites = size(y)[1]
n_surv = size(y)[2]


@model occ(y, n_sites, n_surv) = begin
    p = Array{Real}(undef, n_sites, n_surv)
    ψ = Array{Real}(undef, n_sites)

    ψ_mean ~ Uniform(0,1)
    p_mean ~ Uniform(0,1)

    for i in 1:n_sites
    # There is at least one detection at siteᵢ 
    # we know zᵢ == 1
         if sum(y[i,:]) > 0
             for j in 1:n_surv
                 #ψ[i] = logistic(ψ_mean)
                 #p[i,j] = logistic(p_mean)
                 y[i,j] ~ Bernoulli(p_mean)
                end
            1 ~ Bernoulli(ψ_mean)
         end
    
    # There are no detections at siteᵢ - c
    # zᵢ == 1 (prob ψ)  and wasn't detected (prob 1 - p), 
    # or zᵢ == 0 (prob 1 - ψ)
        if sum(y[i,:]) == 0
            
            # for j in 1:n_surv
            #     p[i,j] = logistic(p_mean)
            # end

            1 ~ Bernoulli(ψ_mean * (1-p_mean)) 
            1 ~ Bernoulli((1 - ψ_mean))
           
        end
    end
end 


p
z_true = sum(z) / length(z)

chains = mapreduce(c -> sample(occ(y, n_sites, n_surv), NUTS(1000, .95), 2000, drop_warmup = false), chainscat, 1:2)



display(chains)

# trace plots and posteriors
plot(chains)

# running average plots
mp = meanplot(chains::Chains)

# plot joint density
post_ψ = chains[:ψ][:, 1]
post_p = chains[:p][:, 1]
jp = marginalkde(post_p, post_ψ)
plot(mp, jp, layout = (1, 2))








function prob_uncaptured(n_sites, n_surv, p, ψ) 
    z = Matrix{Real}(undef, n_sites, n_surv)
    for i in 1:n_sites
        z[i, n_surv] = 1.0;
        for t in 1:n_surv
            z[i, t] = (1 - ψ_mean) + ψ_mean * (1 - p[i, t]);
        end
    end
    return z        
end




