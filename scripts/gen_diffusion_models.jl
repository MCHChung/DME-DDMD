using DrWatson
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

# load data 
#name = readdir(datadir("sims"))[1]
diff_data = wload(datadir("sims", "diff_data2.jld2"))
xs = diff_data["xs"]
ts = diff_data["ts"]
P = diff_data["P"]

# build differential matrices for Galerkin scheme
wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

# specify inputs for algo 
K = 1 # latent space dim 
N = [1 ; 2 ; 3 ; 4 ; 5] # polynomial orders for library of dZ/dt 
θy = [xs xs.^2 xs.^3 xs.^4 xs.^5] # Model terms for Y = Θy(x)Ξy

# sweep model enforcement hyperparameters
λzs = [1e-1, 1e0, 1e1]
λys = [1e-1, 1e0, 1e1]
models = tmi_w_hp_scan(P, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
cz=1e-6, cy=1e1, use_rand_init=false, sparse_iter=50000)

# score models 
ic_scores = score_models(models, N, θy, W, DW)

# save data 
diff_models_and_scores = @strdict λzs λys models ic_scores
wsave(datadir("models", "diff_models2.jld2" ), diff_models_and_scores)