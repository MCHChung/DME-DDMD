using DrWatson, Random
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

# load data 
br_data = wload(datadir("sims", "brownian_data.jld2"))
ts = br_data["ts"]
xs = br_data["xs"]
ys = br_data["ys"]
Xd = br_data["Xd"]
u0 = br_data["u0"]

wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

K = 1 # latent space dim 
N = [1; 2; 3; 4; 5] # polynomial terms for model 
xs_rp = reshape(repeat(xs,1,length(ys))', (length(xs)*length(ys),1))
ys_rp = reshape(repeat(reverse(ys),1,length(xs)), (length(xs)*length(ys),1))
θy = [xs_rp ys_rp xs_rp.^2 xs_rp.*ys_rp ys_rp.^2] # Model terms for Y = θy*Ξy 

# sweep model enforcement hyperparameters
λzs = [1e-1, 1e0, 1e1]
λys = [1e-1, 1e0, 1e1]
Xd_rs = zeros((size(Xd,1)*size(Xd,2), size(Xd,3)))
for i=1:size(Xd,3)
    Xd_rs[:,i] = reshape(Xd[:,:,i], (size(Xd,1)*size(Xd,2),1))
end
Xd_rs = Xd_rs ./ sum(Xd_rs, dims=1)  

#models = tmi_w_hp_scan(Xd_rs, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
#cz=1e-6, cy=1e1, use_rand_init=true, sparse_iter=50000, seed=0)
 
models = []
ic_scores = []
z0s = []
Y0s = []
rng = Random.default_rng()
a, α = size(Xd_rs)
for i=1:10
    println("i => $(i)")

    Random.seed!(rng, i-1)
    z = randn(rng, α, K)
    Y = randn(rng, a, K)
    push!(z0s, z)
    push!(Y0s, Y)
    models_i = tmi_w_hp_scan(Xd_rs, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
    cz=1e-6, cy=1e1, use_rand_init=true, sparse_iter=50000, seed=i-1, z=z, Y=Y)

    ic_score_i = score_models(models_i, N, θy, W, DW)

    push!(models, models_i)
    push!(ic_scores, ic_score_i)
end

# score models 
#ic_scores = score_models(models, N, θy, W, DW)

# save data 
br_models_and_scores = @strdict λzs λys models ic_scores z0s Y0s
wsave(datadir("models", "brownian_models_rand.jld2" ), br_models_and_scores)
