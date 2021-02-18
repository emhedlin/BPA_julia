using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV, SpecialFunctions

#= 
What Happens when p < 1 and constant, and p is not accounted for in a species distribution model

To fully grasph how the site-occupancy model "works", we first look at the simplest possible case: 
both ecological and the observation process are described by an intercept only.
=#

# Spatial and Temporal replication
R, T = 200, 3

# Process Parameters
ψ, p = 0.8, 0.5

# observations
y = zeros(R, T) # empty array
z = rand!(Binomial(1, ψ), empty_array) #fill empty array with latent state

# Observation process
for i in 1:R 
    for t in 1:T 
        y[i,t] = (z[i,t] * rand(Binomial(1, p), 1))[1]
    end
end

sum(z)
sum(y)

# WIP
@model site_occ(y) begin
    n_sites = size(y)[1]
    n_time = size(y)[2]
    occ_obs = zeros(n_sites)
    sum_y = zeros(n_sites)

    for i in 1:n_sites
        sum_y[i] = sum(y[i]);
        if sum_y[i]
            occ_obs = occ_obs + 1
        end
    end

    
    # Likelihood ~~~~~~~
    for i in 1:n_sites
        if sum_y[i] < 0
            1 ~ Bernoulli(psi)
            y[i] ~ Bernoulli(p)
        else
            target += log(sum(exp()))
        end
    end
end


n_sites = size(y)[1]
n_time = size(y)[2]
sum_y = zeros(n_sites)


#= 
The true state of site i, occupied or not, is represented by the 
parameter in our model "z". This is a discrete parameter, either 1 or 0,
which has to be marginalized out to allow HMC sampling. In general,
if you have a joint distribution for y (observation dependent on z and p), 
and z (true state dependent on ψ), we obtain the marginal distribution of y by
summing the joint distribution over all possible values of z, which in This
case is 1 or 0:

marginalized distribution:
                  
                 |log(ψ) + log(Binomial(yi | p))                         1. for yi > 0 
log[y_i | p,ψ] = |
                 |log(e^log(ψ) + log(Binomial(yi | p)) + e^log(1-ψ))     2. for yi = 0

1.
2. log(exp( log(ψ) + log(Binomial(yi|p) ) + exp( log(1-ψ)) ))




=#




x = 1:100
plot(loggamma.(x))
