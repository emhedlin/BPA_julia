using Distributions, Random, Turing, DataFrames, Plots, MCMCChains, StatsPlots, CSV




n_years = 25   # number of years
N1 = 30        # initial population size
mean_λ = 1.02  # mean annula population growth rate
σ²_λ = 0.2     # process (temporal) variation of the growth rate
σ²_y = 20      # variance of the observation error


N = Vector{Float64}(undef, n_years) # this would normally be integer, but BPA generates floats
y = Vector{Float64}(undef, n_years) # this would normally be integer, but BPA generates floats
N[1] = N1
λ = rand(Normal(mean_λ, sqrt(σ²_λ)), n_years-1)

for t in 1:n_years-1
    N[t+1] = N[t] * λ[t]
end

for t in 1:n_years
    y[t] = rand(Normal(N[t], sqrt(σ²_y)),1)[1]
end

plot(1:n_years, N)
plot!(1:n_years, y)


@model simple_model(y): begin
    t = size(y,1)

    mean_λ ~ Uniform(0,10)     # mean growth rate
    σ_proc ~ Uniform(0,10)     # SD of state process
    σ_obs ~ Uniform(0,10)      # SD of observation process
    λ ~ filldist(truncated(Cauchy(0, 2), 0, Inf), t-1)
    
    N̂ = Vector{Float64}(undef, t)
    
    
    # Likelihood

    λ = Normal(mean_λ, σ_proc)

    y ~ Normal(N̂, σ_obs)


end
