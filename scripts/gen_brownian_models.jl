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

models = tmi_w_hp_scan(Xd_rs, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
cz=1e-6, cy=1e1, use_rand_init=false, sparse_iter=50000)
 
# score models 
ic_scores = score_models(models, N, θy, W, DW)

# save data 
br_models_and_scores = @strdict λzs λys models ic_scores
wsave(datadir("models", "brownian_models.jld2" ), br_models_and_scores)
