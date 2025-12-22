using DrWatson
@quickactivate "TMI&MD"

# load src
include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("vis_results.jl"))

# load data 
#name = readdir(datadir("sims"))[1]
ndata = wload(datadir("sims", "neuron_data.jld2"))
ts = ndata["ts"]
P = ndata["Psms_tr"]
P1 = P[1] ./ sum(P[1], dims=1)
P2 = P[2] ./ sum(P[2], dims=1)

K = 2
#=
# hot start 
@time (E1, z1 , Y1, q1) , _ = tmi(P1, K, maxiters=150000, η=1e-3, tol=1e-4, print_err=false, use_E=true)
@time (E2, z2 , Y2, q2) , _ = tmi(P2, K, maxiters=150000, η=1e-3, tol=1e-4, print_err=false, use_E=true)

neuron_hotstart = @strdict E1 E2 z1 z2 Y1 Y2 q1 q2
wsave(datadir("models", "neuron_hotstart.jld2"), neuron_hotstart)
=# 

# build differential matrices for Galerkin scheme
wind = 11
W = BuildWeightMat(ts, wind, 0)
DW = BuildWeightMat(ts, wind, 1)

# specify inputs for algo 
N = [1 0 ; 0 1; 2 0 ; 1 1; 0 2]
cz = 1e-6 
λz = 1e1
@time (E1, z1 , Y1, q1) , Ξz1, _ = tmi(P1, K, W, DW, N, maxiters=150000, η=1e-3, tol=1e-4,
 cz=cz, λz=λz, print_err=false, use_E=true, z=z1, E=E1, Y=Y1, sparse_iter=50000)

@time (E2, z2 , Y2, q2), Ξz2, _  = tmi(P2, K, W, DW, N, maxiters=150000, η=1e-3, tol=1e-4, cz=cz, 
λz=λz, print_err=false, use_E=true , z=z2, E=E2, Y=Y2, sparse_iter=50000)


#=
# sweep model enforcement hyperparameters
λzs = [1e0]
models = tmi_w_hp_scan(P, K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
cz=1e-6,  use_rand_init=false, sparse_iter=50000)
=# 
#=
# score models 
ic_scores = score_models(models, N, θy, W, DW)

# save data 
diff_models_and_scores = @strdict λzs λys models ic_scores
wsave(datadir("models", "diff_models2.jld2" ), diff_models_and_scores)
=# 