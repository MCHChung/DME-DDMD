using DrWatson, Random
@quickactivate "TMI&MD"

# Here you may include files from the source directory
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("get_ens_data.jl"))
include(srcdir("pred_lats.jl"))

# generate data 
# Params
β = 0.4
μ = 1.
σ = 2e-1 
u0 = 0.0
f(u,p,t) = @. β*(μ-u)
g(u,p,t) = σ
dt = 0.01
tspan = (0.0, 7.0) 

rng = Random.default_rng()
Random.seed!(rng, 0)

X, ts = Get1DEnsData(f,g,u0,dt,tspan, rng; num_traj=3000, noise_mag=0.1)

# estimate distribution using KDE
xb = 1.5
xs = -xb:0.01:xb
Xd = EstDist(X,xs)
Xd = Xd ./ sum(Xd, dims=1)

(_, _, q1), _ = tmi(Xd, 1, maxiters=150000, η=1e-3, tol=1e-4, use_E=false)
(_, _, q2), _ = tmi(Xd, 2, maxiters=150000, η=1e-3, tol=1e-4, use_E=false)
(_, _, q3), _ = tmi(Xd, 3, maxiters=150000, η=1e-3, tol=1e-4, use_E=false)

kld(p,q; ϵ=1e-8) = sum(p .* log.((p .+ ϵ) ./ (q .+ ϵ)))

e1 = kld(Xd,q1)
e2 = kld(Xd,q2)
e3 = kld(Xd,q3)