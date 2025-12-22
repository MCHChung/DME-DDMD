using DrWatson
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("pred_lats.jl"))

# load data 
ndata = wload(datadir("sims", "neuron_data3.jld2"))
ts = ndata["ts"]
P = ndata["Psms_tr"]
P1 = P[1] ./ sum(P[1], dims=1)
P2 = P[2] ./ sum(P[2], dims=1)

# specify inputs for algo 
K = 2 # latent space dim 

wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

# specify inputs for algo 
N = [1 0 ; 0 1; 2 0 ; 1 1; 0 2]
cz = 1e-6 

# sweep model enforcement hyperparameters
λzs = [1e-1, 1e0, 1e1]
models_1 = tmi_w_hp_scan(P1, K, W, DW, N, λzs, maxiters=150000, η=1e-3, tol=1e-5,
cz=cz, use_rand_init=true, sparse_iter=50000, use_E=true)
 
models_2 = tmi_w_hp_scan(P2, K, W, DW, N, λzs, maxiters=150000, η=1e-3, tol=1e-5,
cz=cz, use_rand_init=true, sparse_iter=50000, use_E=true)

# score models 
ic_scores_1 = score_models(models_1, N, W, DW)
ic_scores_2 = score_models(models_2, N, W, DW)

# save data 
neuron_models_and_scores = @strdict λzs models_1 models_2 ic_scores_1 ic_scores_2
#wsave(datadir("models", "neuron_models.jld2" ), neuron_models_and_scores)
