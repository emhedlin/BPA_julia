using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV

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
        if sum_y
end