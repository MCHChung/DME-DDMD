using Plots, DrWatson, JLD2 , Lux, LuxCore, Optimisers, Zygote, Printf, Random, TensorOperations

@quickactivate "TMI&MD"

include(srcdir("nnmf.jl"))
include(srcdir("build_weight_mats.jl"))
include(srcdir("train_sindyae.jl"))
include(srcdir("sindyae.jl"))
include(srcdir("libs.jl"))
include(srcdir("pred_lats.jl"))


function sindy_ae_model(sys::Int, in_dim, K::Int; order=2)
    
    @assert 1 <= sys <= 4
    #enc, dec = nothing, nothing
    if sys == 1 || sys == 2
        enc = Chain(
            Dense(in_dim, 100, tanh),
            Dense(100, 25, tanh),
            #Dense(25, 10, tanh),
            Dense(25, K)
        )

        dec = Chain(
            Dense(K, 25, tanh),
            #Dense(10, 25, tanh),
            Dense(25, 100, tanh),
            Dense(100, in_dim),
            Lux.WrappedFunction(softmax)
        )

        snet = SINDyNet([x -> PolyLib(x, order=order, use_cons=false)], [K,K])

        return SINDyAE(enc, dec, snet)
    elseif sys == 3 || sys == 4
         enc = Chain(
            Dense(in_dim, 50, tanh),
            Dense(50, 25, tanh),
            #Dense(25, 10, tanh),
            Dense(25, K)
        )

        dec = Chain(
            Dense(K, 25, tanh),
            #Dense(10, 25, tanh),
            Dense(25, 50, tanh),
            Dense(50, in_dim),
            Lux.WrappedFunction(softmax)
        )

        snet = SINDyNet([x -> PolyLib(x, order=order, use_cons=false)], [K,K])

        return SINDyAE(enc, dec, snet)
    end
     
    #return AE(enc, dec)
end

function kld_loss(p, tstate::Lux.Training.TrainState; eps=1e-8)
    enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
    dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
    q = dec_sf(enc_sf(p))
    return kld_loss(p,q,eps=eps)
end

function sweep_sindyae_models(rng, sys::Int, ts, X, Ks; epochs=10000, wind=11)
    klds_ae = zeros(length(Ks))
    vjp = AutoZygote()
    opt = Adam(0.001f0)
    params = []
    states = []
    D, W = BuildWeightMat(ts, wind, 0), BuildWeightMat(ts, wind, 1)
    for i=1:length(Ks)
        println("K => $(Ks[i])")
        model = sindy_ae_model(sys, size(X,1), Ks[i])
        ps,st = Lux.setup(rng,model)
        tstate = Lux.Training.TrainState(model, ps ,st, opt)
        tstate = train_sindyae(tstate, vjp, (X,X), 0, epochs, D, W)
        push!(params, tstate.parameters)
        push!(states, tstate.states)
        kld = kld_loss(X, tstate)
        klds_ae[i] = kld
    end

    return klds_ae, params, states
end

br_data = wload(datadir("sims", "brownian_data.jld2"))
ou_data = wload(datadir("sims", "ou_data.jld2"))
np_data = wload(datadir("sims", "np_data.jld2"))
neuron_data = wload(datadir("sims", "neuron_data.jld2"))

Ks = collect(1:20)
epochs = 35000
rng = Random.default_rng()
Random.seed!(rng, 0)

# load data
# ======= brownian ========
Xbr = br_data["Xd"]
ts_br = br_data["ts"]
Xbr = reshape(Xbr, (size(Xbr,1)*size(Xbr,2), size(Xbr,3)))
Xbr = Xbr ./ sum(Xbr, dims=1)
# ======= OU =========
Xou = ou_data["Xd"]
ts_ou = ou_data["ts"]
Xou = Xou ./ sum(Xou, dims=1)
# ======= NP =========
Xnp = np_data["Ns"][:, 1:1000]
ts_np = np_data["ts"][1:1000]
Xnp = Xnp ./ sum(Xnp, dims=1)
# ======= neuron =========
P = neuron_data["Psms_tr"]
ts_ne = neuron_data["ts"]
Xne1, Xne2 = abs.(P[1])[:,:,1:3], abs.(P[2])[:,:,1:3]
Xne1 = Xne1 ./ sum(Xne1, dims=1)
Xne2 = Xne2 ./ sum(Xne2, dims=1)

# sweep models
println("Brownian")
klds_br, params_br, states_br  = sweep_sindyae_models(rng, 1, ts_br, Xbr, Ks; epochs=epochs)
println("OU")
klds_ou, params_ou, states_ou = sweep_sindyae_models(rng, 2, ts_ou, Xou, Ks; epochs=epochs)
println("NP")
klds_np, params_np, states_np = sweep_sindyae_models(rng, 3, ts_np, Xnp, Ks; epochs=epochs)
println("Neuron 1")
klds_ne1, params_ne1, states_ne1 = sweep_sindyae_models(rng, 4, ts_ne, Xne1, Ks; epochs=epochs)
println("Neuron 2")
klds_ne2, params_ne2, states_ne2 = sweep_sindyae_models(rng, 4 , ts_ne, Xne2, Ks; epochs=epochs)

