using DrWatson, Random
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

# load data 
#name = readdir(datadir("sims"))[3]
ou_data = wload(datadir("sims", "ou_data.jld2"))
ts = ou_data["ts"]
xs = ou_data["xs"]
Xd = ou_data["Xd"]
u0 = ou_data["u0"]

wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

K = 2 # latent space dim 
N = [1 0 ; 0 1 ; 2 0 ; 1 1 ; 0 2] # polynomial terms for model 
xn = xs #.- u0
θy = [xn xn.^2 xn.^3 xn.^4 xn.^5] # Model terms for Y = θy*Ξy 

# sweep model enforcement hyperparameters
λzs = [1e-1, 1e0, 1e1]
λys = [1e-1, 1e0, 1e1]   
models = tmi_w_hp_scan(Xd, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
cz=1e-6, cy=1e1, use_rand_init=false, sparse_iter=50000)

# score models 
ic_scores = score_models(models, N, θy, W, DW)

# save data 
ou_models_and_scores = @strdict λzs λys models ic_scores
#wsave(datadir("models", "ou_models.jld2" ), ou_models_and_scores)
