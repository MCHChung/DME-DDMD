using DrWatson, Random
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

np_data = wload(datadir("sims", "np_data.jld2"))
ts = np_data["ts"]
Ns = np_data["Ns"]
nmax = np_data["nmax"]

Ns_n = Ns ./ sum(Ns, dims=1)

wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

K = 1 # latent space dim 
N = [1 ; 2 ; 3 ; 4 ; 5] # polynomial terms for model 
ns = 1:nmax
θy = [ns ns.^2 ns.^3 ns.^4 ns.^5] # Model terms for Y = θy*Ξy

# sweep model enforcement hyperparameters
λzs = [1e-1, 1e0, 1e1]
λys = [1e-1, 1e0, 1e1]
#models = tmi_w_hp_scan(Ns_n, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
#cz=1e-6, cy=1e1, use_rand_init=false, sparse_iter=50000)

models = []
z0s = []
Y0s = []
ic_scores = []
rng = Random.default_rng()
a, α = size(Ns_n)
for i=1:10
    println("i => $(i)")
    
    Random.seed!(rng, i-1)
    z = randn(rng, α, K)
    push!(z0s, z)

    Y = randn(rng, a, K)
    push!(Y0s, Y)

    models_i = tmi_w_hp_scan(Ns_n, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
    cz=1e-6, cy=1e1, use_rand_init=true, sparse_iter=50000, seed=i-1, z=z, Y=Y)

    ic_score_i = score_models(models_i, N, θy, W, DW)

    push!(models, models_i)
    push!(ic_scores, ic_score_i)
end

# score models 
#ic_scores = score_models(models, N, θy, W, DW)

# save data 
np_models_and_scores = @strdict λzs λys models ic_scores z0s Y0s
wsave(datadir("models", "np_models_rand.jld2" ), np_models_and_scores)
