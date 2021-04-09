

# Chapter 7 - CJS ~~~~~~~~~~~~



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






# Chapter 13 Occupancy ~~~~~~~~~~~~



function occ_sim(N,J,ψ,p) 
        # true state
    z = zeros(N) # empty array
    z = rand!(Binomial(1, ψ), z) #fill empty array with latent state
    z_mat = z .* ones(N, J) # replicate occupancy state across surveys
    y = zeros(N,J)
    # Observation process
    for i in 1:N 
        for t in 1:J 
            y[i,t] = (z_mat[i,t] * rand(Binomial(1, p), 1))[1]
        end
    end
    return y, z_mat
end



function sim_occ_cov(n_sites, n_surv, xmin, xmax, α_ψ, β_ψ, α_p, β_p)
    ψ = Array{Real}(undef, n_sites)
    y = Array{Any}(undef, n_sites, n_surv)
    z = Array{Any}(undef, n_sites)

    X = rand(Uniform(xmin, xmax), n_sites)
    ψ = logistic.(α_ψ .+ β_ψ.* X)
    z = map(rand, Binomial.(1, ψ))    
    occ_fs = sum(z)

    p = logistic.(α_p .+ β_p.*X)
    
    # survey effect
    p_eff = z.*p
    for t in 1:n_surv
        y[:,t] = map(rand, Binomial.(1, p_eff))
    end
    
    return y, z, X
end





function sim_occ_dyn(n_sites, n_surv, n_years, ψ₁, range_p, range_ϕ, range_γ)
    #= 
    Annual variation in probabilities of patch survival, colonization, and detection
    is specified by the bounds of a uniform distribution.
    range_p = bounds of uniform distribution from which annual p drawn
    range_ϕ = bounds of uniform dist. from which annual survival and extinction (1-ϕ) are drawn from
    range_γ = same as range_ϕ and range_p
    =#

    site = 1:1:n_sites
    year = 1:1:n_years
    
    ψ = Array{Real}(undef, n_years)
    μ_z = Array{Real}(undef, n_sites, n_years)
    z = Array{Real}(undef, n_sites, n_years)
    y = Array{Real}(undef, n_sites, n_surv, n_years)
    ψ[1] = ψ₁
    p = rand(Uniform(range_p[1], range_p[2]), n_years)
    ϕ = rand(Uniform(range_ϕ[1], range_ϕ[2]), n_years-1)
    γ = rand(Uniform(range_γ[1], range_γ[2]), n_years-1)
    
    # first year occupancy
    z[:,1] = map(rand, Binomial.(ones(n_sites), repeat([ψ[1]], n_sites)))

    # following years
    for i in 1:n_sites
        for t in 2:n_years
            μ_z[t] = z[i,t-1] .* ϕ[t-1] .+ (1 .- z[i,t-1]) .* γ[t-1]
            z[i,t] = rand(Binomial.(1, μ_z[t]))
        end
    end
    # Observations 
    for i in 1:n_sites
        for t in 1:n_years 
            for j in 1:n_surv
                y[i,j,t] = rand(Binomial(1, z[i,t] * p[t]))
            end
        end
    end

    return y, γ, ϕ, p, z
end
