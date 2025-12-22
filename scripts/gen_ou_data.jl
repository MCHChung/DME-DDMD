using DrWatson, Random
@quickactivate "TMI&MD"

# Here you may include files from the source directory
include(srcdir("get_ens_data.jl"))

# ============= generate train data =============
# Params
β = 0.4
μ = 0.7
σ = 2e-1 
u0 = 0.
f(u,p,t) = @. β*(μ-u)
g(u,p,t) = σ
dt = 0.01
tspan = (0.0, 8.0) 

rng = Random.default_rng()
Random.seed!(rng, 0)

X, ts = Get1DEnsData(f,g,u0,dt,tspan, rng; num_traj=3000, noise_mag=0.05)

# estimate distribution using KDE
xb = 1.5
xs = -xb:0.01:xb
# train data 
Xd = EstDist(X,xs)
Xd = Xd ./ sum(Xd, dims=1)

# create and save data 
ou_data = @strdict ts xs X Xd 
#wsave(datadir("sims", "ou_data.jld2"), ou_data)


# ============ generate test data ===============

β = 0.4
μ = 0.7
σ = 2e-1 
u0s = rand(rng, -1:0.1:0.5, 10) # random ICs
f(u,p,t) = @. β*(μ-u)
g(u,p,t) = σ
dt = 0.01
tspan = (0.0, 8.0) 

Xs = []
Xds = []
for i=1:length(u0s)
    u0_new = u0s[i]
    X_new, _ = Get1DEnsData(f,g, u0_new, dt,tspan, rng; num_traj=3000, noise_mag=0.05)
    Xd_new = EstDist(X_new,xs)
    Xd_new = Xd_new ./ sum(Xd, dims=1)
    push!(Xs, X_new)
    push!(Xds, Xd_new)
end

ou_test_data = @strdict ts xs Xs Xds u0s
wsave(datadir("sims", "ou_test_data.jld2"), ou_test_data)