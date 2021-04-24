using Distributions, Random, Turing, DataFrames, Plots, MCMCChains, StatsPlots, CSV

include("plot_utils.jl")



n_years = 25   # number of years
N₁ = 30        # initial population size
mean_λ = 1.02  # mean annula population growth rate
σ²_λ = 0.02     # process (temporal) variation of the growth rate
σ²_y = 20      # variance of the observation error


N = Vector{Float64}(undef, n_years) # this would normally be integer, but BPA generates floats
y = Vector{Float64}(undef, n_years) # this would normally be integer, but BPA generates floats
N[1] = N₁
λ = rand(Normal(mean_λ, sqrt(σ²_λ)), n_years-1)


for t in 2:n_years
    N[t] = N[t-1] * λ[t-1]
end 
scatter(1:n_years, N; defs2 ..., markersize = 4)
plot!(1:n_years, N; defs1 ..., seriescolor = primary, lw = 1)

for t in 1:n_years
    y[t] = rand(Normal(N[t], sqrt(σ²_y)), 1)[1]
end

# true values versus observed   
scatter!(1:n_years, y;defs1 ..., markersize = 4)
plot!(1:n_years, y; defs1 ..., seriescolor = grey1, lw = 1)


# state
# Nₜ₊₁ = Nₜ * λₜ  - population size at time t+1 = the population size at time t * growth rate
# λₜ ~ Normal(λ̄, σ²_λ) - The time specific growth rates are realizations of a normal random process with mean λ̄ and variance σ²_λ
# Nₜ where t = 1 is not defined, and needs to be specified with a prior, or the raw count

# observation
# yₜ = Nₜ + ϵₜ 
# ϵₜ ~ Normal(0, σ²_y)

length(y)


@model simple_model(y)= begin
    T = length(y)

    λ̄   ~ Uniform(0,10) # mean growth rate
    σ_λ ~ Uniform(0,10) # SD of state process
    σ_y ~ Uniform(0,10) # SD of observation process
        
    λ = Vector{Real}(undef, T-1)
    N̂ = Vector{Real}(undef, T)
    
    σ²_λ = σ_λ^2
    σ²_y = σ_y^2
    
    #for t in 1:T-1
    #    N̂[t+1] = N̂[t] * λ[t]
    #end
    N̂[1] ~ Uniform(0,500)
    

    # Likelihood
    
    for t in 1:T
        if t == 1
            λ[t] ~ Normal(λ̄, σ²_λ)
            y[t] ~ Normal(N̂[t], σ²_y)
        else
            λ[t] ~ Normal(λ̄, σ²_λ)
            y[t] ~ Normal(N̂[t-1] * λ[t], σ²_y)
        end
    end

    # N 10 11 12 13
    # y 10 10 10 15 
    # λ .1 .1 .1
    
    # for t in 1:T-1
    #     N̂[t+1] = N̂[t] * λ[t]
    #     λ[t] ~ Normal(λ̄, σ²_λ)
    # end
    # 
    # for t in 1:T
    #     y[t] ~ Normal(N̂[t], σ²_y)
    # end
    # use generated_quantities(simple_model, chain)
    σ²_λ = σ_λ^2
    σ²_y = σ_y^2
    return σ²_λ, σ²_y
    #λ .~ Normal.(mean_λ, σ_proc)
    #y .~ Normal.(N̂, σ_obs)
end

chain = sample(simple_model(y),  NUTS(1000, 0.65), 1000, drop_warmup = false)







# state
# Nₜ₊₁ = Nₜ * λₜ  - population size at time t+1 = the population size at time t * growth rate
# λₜ ~ Normal(λ̄, σ²_λ) - The time specific growth rates are realizations of a normal random process with mean λ̄ and variance σ²_λ
# Nₜ where t = 1 is not defined, and needs to be specified with a prior, or the raw count

# observation
# yₜ = Nₜ + ϵₜ 
# ϵₜ ~ Normal(0, σ²_y)



