using DrWatson, Plots, LinearAlgebra , KernelDensity, StatsBase
@quickactivate 

include(srcdir("tmi.jl"))
include(srcdir("build_weight_mats.jl"))


function HK_Dynamics(n::Int, ϵ::Float64; maxiters=1000, X=nothing, numgridpts=100)
    # set initial opinions
    X = (X == nothing) ? X = rand(n) : X 
    Xs = zeros(n, maxiters)
    Xds = zeros(numgridpts, maxiters)
    Xs[:,1] = X
    Xds[:,1] = get_density(X, numgridpts=numgridpts)

    for iter=2:maxiters 
        Xtmp = copy(X)
        @simd for i=1:n 
            inds = findall(y -> y <= ϵ , abs.(X .- X[i]))
            Xtmp[i] = mean(X[inds])
        end

        X = Xtmp
        Xs[:, iter] = X
        Xds[:,iter] = get_density(X, numgridpts=numgridpts)
    end

    return Xs , Xds
end

function get_density(X; numgridpts=100)
    xs = range(0,1,numgridpts)
    k = kde(X)
    Xd = pdf(k, xs)
    return Xd
end

function Ensemble_HK(numruns::Int, n::Int, ϵ::Float64, σ::Float64;maxiters=100, numgridpts=250)
    Xds = zeros((numgridpts, maxiters, numruns))
    for run = 1:numruns
        X0 = 0.5 .+ σ*randn(n)  
        _ , Xd = HK_Dynamics(n, ϵ, maxiters=maxiters, X=X0, numgridpts=numgridpts)
        Xds[:,:,run] = Xd
    end

    return Xds
end

function Ensemble_HK(numruns::Int, n::Int, ϵ::Float64;maxiters=100, numgridpts=250)
    Xds = zeros((numgridpts, maxiters, numruns))
    for run = 1:numruns
        _ , Xd = HK_Dynamics(n, ϵ, maxiters=maxiters, X=nothing, numgridpts=numgridpts)
        Xds[:,:,run] = Xd
    end

    return Xds
end

#=
n = 2500 
ϵ = 0.05 
X0 = randn(n)*0.3 .+ 0.5
maxiters=100
numgridpts = 500
Xs , Xds = HK_Dynamics(n, ϵ, maxiters=maxiters, X=X0, numgridpts=numgridpts)

xs = range(0,1,numgridpts) 
ts = 0:maxiters-1
heatmap(ts, xs, Xds)
=# 

numruns = 9
n = 10000 
ϵ = 0.05 
σ = 0.4
maxiters=100
numgridpts = 500
Xds = Ensemble_HK(numruns, n, ϵ, σ ; maxiters=maxiters, numgridpts=numgridpts)

xs = range(0,1,numgridpts) 
ts = 0:maxiters-1
ps = []
for i=1:9
    p = heatmap(ts,xs,Xds[:,:,i])
    push!(ps,p)
end

plot(ps..., layout=(3,3), size=(900,500))

#=
Ks = collect(1:5)
qs_cp, klds_cp = latent_dim_scan(Xds[:,:,1], Ks, normalize=true ,tol =1e-5, η=1e-3, maxiters=100000, use_E=false)
scatter(klds_cp)
=# 

DW, W = BuildDiscWeightMat(ts)

K = 1 # latent space dim 
N = [1 ; 2 ; 3 ; 4 ; 5] # polynomial terms for model 
θy = [xs xs.^2 xs.^3 xs.^4 xs.^5] # Model terms for Y = θy*Ξy

# sweep model enforcement hyperparameters
λzs = [1e-1, 1e0, 1e1]
λys = [1e-1, 1e0, 1e1]
models = tmi_w_hp_scan(Xds[:,:,1], K, W, DW, N, θy, λzs, λys, maxiters=150000, η=1e-3, tol=1e-4,
cz=1e-6, cy=1e1, use_rand_init=false, sparse_iter=50000)

# score models 
ic_scores = score_models(models, N, θy, W, DW)