using Distributions, Random, Turing, MCMCChains, Gadfly
using StatsFuns: logistic




# Single Season Occupancy, constant parameters

# Spatial and Temporal replication
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



# Generate Data for single year occupancy with constant parameters
y, z_mat = occ_sim(500, 3, 0.6, 0.4)

n_sites = size(y)[1]
n_surv = size(y)[2]

# True estimates (some noise around inputs)
ψ_true = sum(z_mat[:,1]) / n_sites
p_true = sum(y) / sum(z_mat)

@model occ(y, n_sites, n_surv) = begin

    # Priors ~~~
    ψ_mean ~ Truncated(Normal(0.5, 0.2), 0,1)
    p_mean ~ Truncated(Normal(0.5, 0.2), 0,1)

    # Likelihood ~~~
    for i in 1:n_sites
        if sum(y[i,:]) > 0
            for j in 1:n_surv
                 y[i,j] ~ Bernoulli(p_mean)
            end
            1 ~ Bernoulli(ψ_mean)
         end
        if sum(y[i,:]) == 0
        1 ~ Bernoulli( (ψ_mean*(1-p_mean)^n_surv) + (1-ψ_mean) ) 
        end
    end
end 

n_sites, n_surv = size(y)

chains = mapreduce(c -> sample(occ(y, n_sites, n_surv), NUTS(1000, .65), 4000, drop_warmup = false), chainscat, 1:2)
# chain = sample(occ(y, n_sites, n_surv),  PG(20), 1000, save_state = false)

# trace plots and posteriors
plot(chains)
p_true
ψ_true

# running average plots
mp = meanplot(chains::Chains)

# plot joint density
ψ_post = chains[:ψ_mean][:, 1]
p_post = chains[:p_mean][:, 1]
joint = marginalkde(post_p, post_ψ)
plot(mp, jp, layout = (1, 2))

# Simulate multiple data sets with all 
# combinations of ψ and p between 0.3 and 0.7

ψ_range = collect(0.3:0.2:0.7)
p_range = collect(0.3:0.2:0.7)
x = collect(Base.product(ψ_range,p_range))
fill = Array{Real}(undef, length(x), 4)

for i in 1:length(x)  
        y, z_mat = occ_sim(500,5,x[i][1],x[i][2]) # N,J,ψᵢ,pᵢ 
        chains = sample(occ(y, 500, 5), NUTS(1000, .95), 1000, drop_warmup = false)
        fill[i,1] = x[i][1]
        fill[i,2] = x[i][2]
        fill[i,3] = mean(chains[:ψ_mean])
        fill[i,4] = mean(chains[:p_mean])
end

true_ψ = fill[:,1]
true_p = fill[:,2]
m_ψ = fill[:,3]
m_p = fill[:,4]



plot(layer(x = 1:1:9, y=true_ψ, Geom.point),
     layer(x = 1:1:9, y=m_ψ, Geom.point),
     grid_color = "white"
     )


     plot!(x = fill[:,2], y = fill[:,4])


plot!(1:81, fill[:,3], seriestype = :scatter, palette = :Blues_9)


plot(1:81, fill[:,2], seriestype = :scatter, seriesalpha = 0.25)
plot!(1:81, fill[:,4], seriestype = :scatter)

