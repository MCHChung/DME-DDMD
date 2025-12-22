using Distributions, StatsBase , Random
using FFTW, ImageFiltering


function GenerateSpikingDataWithTrials(Xs::Array{Float64,3}, trind::Int, dt; trials=100, N=200, seed=0, σ=0.2)
    
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    T = size(Xs,2)
    C = randn(rng, N, size(Xs,1)) 
    b = randn(rng, N)

    ics = size(Xs,3)
    Ps = zeros(N, T, ics, trials)

    for ic=1:ics
        println("ic => $(ic)")
        X = Xs[:,:,ic]
        λs = exp.(C*X .+ b) # firing rates 
        #λs = λs ./ sum(λs, dims=1)
        for tr=1:trials
            for i=1:T
                for n=1:N
                    d = Poisson(λs[n,i]*dt)
                    Ps[n,i,ic,tr] = rand(rng, d)
                end
            end
        end
    end

    Pavg_tr = mean(Ps[:,:,1:trind,:], dims=4)[:,:,:,1]
    Pavg_test = mean(Ps[:,:,trind+1:end,:], dims=4)[:,:,:,1]
    
    Psm_tr = copy(Pavg_tr)
    Psm_test = copy(Pavg_test)

    Psm_tr = GaussSmData(Psm_tr; σ=σ)
    Psm_test = GaussSmData(Psm_test; σ=σ)

    return (C,b), Ps, (Pavg_tr, Pavg_test), (Psm_tr, Psm_test)
end


function GaussSmData(X; σ=0.2)
    kern = ImageFiltering.Kernel.gaussian((σ, 0))[:]
    Xsm = similar(X)
    if size(X,3) != 1
        for tr=1:size(X,3)
            for n=1:size(X,1)
                Xsm[n,:,tr] = imfilter(X[n,:,tr], kern)
            end
        end
    else 
        for n=1:size(X,1)
            Xsm[n,:] = imfilter(X[n,:], kern)
        end
    end 

    return Xsm
end