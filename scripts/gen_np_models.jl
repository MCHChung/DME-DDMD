using DrWatson, Random
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

np_data = wload(datadir("sims", "np_data2.jld2"))
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
models = tmi_w_hp_scan(Ns_n, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
cz=1e-6, cy=1e1, use_rand_init=false, sparse_iter=50000)

# score models 
ic_scores = score_models(models, N, θy, W, DW)

# save data 
np_models_and_scores = @strdict λzs λys models ic_scores
wsave(datadir("models", "np_models2.jld2" ), np_models_and_scores)
