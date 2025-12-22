using DrWatson, Random
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))
include(srcdir("smooth.jl"))

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
#λzs = [1e1]
#λys = [1e1] 
#models = tmi_w_hp_scan(Xd, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
#cz=1e-6, cy=1e1, use_rand_init=true, sparse_iter=50000)

models = []
z0s = []
Y0s = []
ic_scores = []
inds = []
U , _ , V = svd(log.(Xd .+ 1e-6))
a, α = size(Xd)
rng = Random.default_rng()
for i=1:10
    println("i => $(i)")
    Random.seed!(rng, i-1)

    z = randn(rng, α, K)
    push!(z0s, z)

    Y = randn(rng, a, K)
    push!(Y0s, Y)
    
    models_i = tmi_w_hp_scan(Xd, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
    cz=1e-6, cy=1e1, use_rand_init=true, sparse_iter=50000, seed=i-1, z=z, Y=Y)

    ic_score_i = score_models(models_i, N, θy, W, DW)

    push!(models, models_i)
    push!(ic_scores, ic_score_i)
end

#VisModel(models[1][findmin(ic_scores[1])[2]]["Ξz"],"")
# score models 
#ic_scores = score_models(models, N, θy, W, DW)

# save data 
ou_models_and_scores = @strdict λzs λys models ic_scores z0s Y0s
wsave(datadir("models", "ou_models_rand_2.jld2" ), ou_models_and_scores)
