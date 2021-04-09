using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV
using StatsFuns: logistic

#= 
single season occupancy with covariates
=#


y, z, X = sim_occ_cov(200, 3, -1, 1, -1, 3, 1, -3)


# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@model occ(y,n_sites, n_surv, X) = begin
    p = Array{Real}(undef, n_sites, n_surv)
    ψ = Array{Real}(undef, n_sites)
    

    # Priors
    for i in 1:length(n_sites)
        ψ[i] ~ Uniform(0,1)
        for j in 1:length(n_surv)
            p[i,j] ~ Uniform(0,1)
        end
    end
    #ψ ~ filldist(Uniform(0,1), n_sites)
    α_ψ ~ Normal(0,4)
    β_ψ ~ Normal(0,4)
    α_p ~ Normal(0,4)
    β_p ~ Normal(0,4)



    # Likelihood    
    for i in 1:n_sites
    # There is at least one detection at siteᵢ 
    # we know zᵢ == 1
        if sum(y[i,:]) > 0
            for j in 1:n_surv # ADD BROADCAST 
                p[i,j] = logistic( α_p + β_p* X[i] )
                y[i,j] ~ Bernoulli(p[i,j])
            end
            ψ[i] = logistic( α_ψ + β_ψ*X[i] )
            1 ~ Bernoulli(ψ[i])
        end
    
    # There are no detections at siteᵢ - c
    # zᵢ == 1 (prob ψ)  and wasn't detected (prob 1 - p), 
    # or zᵢ == 0 (prob 1 - ψ)
        if sum(y[i,:]) == 0
            
            for j in 1:n_surv # ADD BROADCAST
                p[i,j] = logistic(α_p + β_p*X[i])
            end 
           
            ψ[i] = logistic( α_ψ + β_ψ*X[i] )
            1 ~ Bernoulli( (ψ[i]*( (1-p[i,1])*(1-p[i,2])*(1-p[i,3]) )) + (1-ψ[i]) )

        end
    end
end 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# chains = mapreduce(c -> sample(occ(y, n_sites, n_surv), NUTS(1000, .95), 2000, drop_warmup = false), chainscat, 1:2)

n_sites, n_surv = size(y)
chains = sample(occ(y, n_sites, n_surv, X), NUTS(1000, .65), 2000, drop_warmup = false)


chains = mapreduce(c -> sample(occ(y, n_sites, n_surv), NUTS(1000, .65), 2000, drop_warmup = false), chainscat, 1:2)

α_p
α_ψ
β_p
β_ψ


# trace plots and posteriors
plot(chains)
α_p
α_ψ
β_p
β_ψ


# running average plots
mp = meanplot(chains::Chains)

# plot joint density
post_ψ = chains[:ψ][:, 1]
post_p = chains[:p][:, 1]
jp = marginalkde(post_p, post_ψ)
plot(mp, jp, layout = (1, 2))


