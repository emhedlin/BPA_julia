using Distributions, Random, Turing, DataFrames, Distributed, Plots, MCMCChains, StatsPlots, CSV
using StatsFuns: logistic
include("utils.jl")

#= 
Dynamic multi-season occupancy 

Unmarginalized~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z₁ ~ Bernoulli(ψ₁)                Season 1
zₖ₊₁|zₖ ~ Bernoulli(zₖϕₖ + (1 - zₖ)γₖ)  Markovian transitions in following seasons
  yₜ|zₜ ~ Bernoulli(zₜp + (1 - zₜ)q)    Observation process

ψ₁ = initial occupancy probability
ϕₖ = survival probability
γₖ = colonization probability

Marginalization ~~~~~~~~~~~~~~~~~~~~~~~~~~
Potential state transitions

zₜ = 1 | zₜ₋₁ = 1 ~ Bernoulli(ϕ)   # survival
zₜ = 0 | zₜ₋₁ = 1 ~ Bernoulli(1-ϕ) # extinction
zₜ = 1 | zₜ₋₁ = 0 ~ Bernoulli(γ)   # colonized
zₜ = 0 | zₜ₋₁ = 0 ~ Bernoulli(1-γ) # remains unoccupied

yₜ = 1 | yₜ₋₁ = 1 # Survival
y[i,j] ~ Bernoulli(p[i,j])
1 ~ Bernoulli(ϕ)

yₜ = 1 | yₜ₋₁ = 0 # Colonization or Survival
    y[i,j,k] ~ Bernoulli(p_mean)
 1 ~ Bernoulli( γ )
 1 ~ Bernoulli(ϕ * (p)^n_surv))


yₜ = 0 | yₜ₋₁ = 1 # non-survival or Survival
1 ~ Bernoulli( (1-ϕ * (p)^n_surv) + ϕ )

yₜ = 0 | yₜ₋₁ = 0 # Extinction or Survival or Colonization or remains unoccupied
1 ~ Bernoulli( 
    (1-ϕ * (p^n_surv) * (1-p)^n_surv) +   # Extinction:        zₜ = 0 and zₜ₋₁ = 1
    (ϕ * (1-p)^n_surv * (1-p)^n_surv) +   # Survival:          zₜ = 1 and zₜ₋₁ = 1
    (γ * (1-p)^n_surv * (p)^n_surv) +     # Colonization       zₜ = 1 and zₜ₋₁ = 0
    (1-γ * (p^n_surv) * p^nsurv)          # remains unoccupied zₜ = 1 and zₜ₋₁ = 0
)
=#

# γ_true = Array{Real}(undef, n_years-1)
# ϕ_true = Array{Real}(undef, n_years-1)
# p_true = Array{Real}(undef, n_years)

#                                            N   J   K   ψ₁       p          ϕ         γ 
y, γ_true, ϕ_true, p_true, z = sim_occ_dyn( 250, 3, 10, 0.4, [0.2 0.4], [0.6 0.8], [0.3 0.6] )


# Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@model dynocc(y, n_sites, n_surv, n_years) = begin

    γ = Array{Real}(undef, n_years-1)
    ϕ = Array{Real}(undef, n_years-1)
    p = Array{Real}(undef, n_years)

    # Priors ~~~~~~~~~~~~~~~~~~~~~~~~~~
    ψ₁ ~ Beta(1,1)
    γ ~ filldist(Beta(1,1), n_years-1)
    ϕ ~ filldist(Beta(1,1), n_years-1)
    p ~ filldist(Beta(1,1), n_years)
    


    # Likelihood ~~~~~~~~~~~~~~~~~~~~~~~~
    for i in 1:n_sites
        
        # occupancy in the first year
        if sum(y[i,:,1]) > 0
            for j in 1:n_surv
                y[i,j,1] ~ Bernoulli(p[1])
            end
            1 ~ Bernoulli(ψ₁)
        end
        if sum(y[i,:,1]) == 0
            1 ~ Bernoulli( (ψ₁*((1-p[1])^n_surv)) + (1-ψ₁) ) 
        end
    end

        # Markovian process for following years
    for i in 1:n_sites    
        for k in 2:n_years
            if sum(y[i,:,k]) > 0
                if sum(y[i,:,k-1]) > 0                                  # yₜ = 1 | yₜ₋₁ = 1
                    for j in 1:n_surv
                        y[i,j,k] ~ Bernoulli(p[k])
                    end
                    1 ~ Bernoulli(ϕ[k-1])
                end
            if sum(y[i,:,k]) == 0                                       # yₜ = 1 | yₜ₋₁ = 0  
                for j in 1:n_surv
                    y[i,j,k] ~ Bernoulli(p[k])
                end
                1 ~ Bernoulli( (γ[k-1]) +                                         # Colonization       zₜ = 1 and zₜ₋₁ = 0
                               (ϕ[k-1] * ((1-p[k-1])^n_surv)) )                   # Survival:          zₜ = 1 and zₜ₋₁ = 1
                end
                
            end
            if sum(y[i,:,k]) == 0
                if sum(y[i,:,k-1]) > 0 #                                # yₜ = 0 | yₜ₋₁ = 1                            
                    1 ~ Bernoulli( ((1-ϕ[k-1]) +                                    # Extinction:        zₜ = 0 and zₜ₋₁ = 1 
                                   (ϕ[k-1] * ((1-p[k])^n_surv))))                   # Survival:          zₜ = 1 and zₜ₋₁ = 1
                end
                # println("ϕ[k-1]= ",ϕ[k-1])
                # println("p[k-1]=",p[k-1])
                # println("p[k]=",p[k])
                # println("γ[k-1]=",γ[k-1])
                if sum(y[i,:,k-1]) == 0                                 # yₜ = 0 | yₜ₋₁ = 0  
                   1 ~ Bernoulli( ( (1-ϕ[k-1]) * ((1-p[k-1])^n_surv) ) +                   # Extinction:        zₜ = 0 and zₜ₋₁ = 1
                                  ( ϕ[k-1] * ((1-p[k])^n_surv) * ((1-p[k-1])^n_surv) ) +   # Survival:          zₜ = 1 and zₜ₋₁ = 1
                                  (γ[k-1] * ((1-p[k])^n_surv)) +                           # Colonization       zₜ = 1 and zₜ₋₁ = 0
                                  (1-γ[k-1]))                                              # remains unoccupied zₜ = 0 and zₜ₋₁ = 0
                end
            end
        end
    end
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# chains = mapreduce(c -> sample(occ(y, n_sites, n_surv), NUTS(1000, .95), 2000, drop_warmup = false), chainscat, 1:2)

n_sites, n_surv, n_years = size(y)
chains = sample(dynocc(y, n_sites, n_surv, n_years), NUTS(500, .65), 1500, drop_warmup = false)
#chains = mapreduce(c -> sample(dynocc(y, n_sites, n_surv, n_years), NUTS(1000, .65), 1000, drop_warmup = false), chainscat, 1:2)

display(chains)
sum(y) / sum(z+z+z)

for i in 1:n_years
    println(sum(y[:,:,i]) / sum(z[:,i] + z[:,i] + z[:,i]))
end
p_true
sum(z[:,1]) / size(z)[1]

R(r::Float64) = (n) → n*r
K(k::Float64) = (n) → n*(1.0-n/k)
p_true


growth = (R(2.3)∘K(1.0))
growth(1.0)

# trace plots and posteriors
plot(chains)

# running average plots
mp = meanplot(chains::Chains)

# plot joint density
mean(chains[:ψ₁])
mean(chains[:γ])
chains

post_p = chains[:p][:, 1]
jp = marginalkde(post_p, post_ψ)
plot(mp, jp, layout = (1, 2))


p

ϕ= 0.9470784983584557
p=0.3937702046769489
p=0.20241700674358212
γ=0.10458884763176284
n_surv = 3




( (1-ϕ) * ((1-p)^n_surv) ) +                   # Extinction:        zₜ = 0 and zₜ₋₁ = 1
( ϕ * ((1-p)^n_surv) * ((1-p)^n_surv) ) +   # Survival:          zₜ = 1 and zₜ₋₁ = 1
(γ * ((1-p)^n_surv)) +                         # Colonization       zₜ = 1 and zₜ₋₁ = 0
(1-γ) * p                                       # remains unoccupied zₜ = 0 and zₜ₋₁ = 0
