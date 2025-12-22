using DrWatson
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("pred_lats.jl"))

# load data 
ndata = wload(datadir("sims", "neuron_data.jld2"))
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
models_1 = []
models_2 = []
ic_scores_1 = []
ic_scores_2 = []
rng = Random.default_rng()
z0s, Y0s, E0s = [], [], []
a, α, T = size(P1)
for i=1:10
    println("i => $(i)")
    Random.seed!(rng, i-1)

    z = randn(rng, α, K, T)
    push!(z0s, z)
    Y = randn(rng, a, K)
    push!(Y0s, Y)
    E = randn(rng, a)
    push!(E0s, E)

    println("Model 1:")
    models_1_i = tmi_w_hp_scan(P1, K, W, DW, N, λzs, maxiters=150000, η=1e-3, tol=1e-5,
    cz=cz, use_rand_init=true, sparse_iter=50000, use_E=true, z=z, Y=Y, E=E)
    
    println("Model 2:")
    models_2_i = tmi_w_hp_scan(P2, K, W, DW, N, λzs, maxiters=150000, η=1e-3, tol=1e-5,
    cz=cz, use_rand_init=true, sparse_iter=50000, use_E=true, z=z, Y=Y, E=E)
    
    # score models 
    ic_scores_1_i = score_models(models_1_i, N, W, DW)
    ic_scores_2_i = score_models(models_2_i, N, W, DW)

    push!(models_1, models_1_i)
    push!(models_2, models_2_i)
    push!(ic_scores_1, ic_scores_1_i)
    push!(ic_scores_2, ic_scores_2_i)
end
# save data 
neuron_models_and_scores = @strdict λzs models_1 models_2 ic_scores_1 ic_scores_2 z0s Y0s E0s
wsave(datadir("models", "neuron_models_rand.jld2" ), neuron_models_and_scores)