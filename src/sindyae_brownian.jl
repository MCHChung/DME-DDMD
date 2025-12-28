using Plots, DrWatson, JLD2 

@quickactivate "TMI&MD"

include(srcdir("build_weight_mats.jl"))
include(srcdir("train_sindyae.jl"))
include(srcdir("sindyae.jl"))
include(srcdir("libs.jl"))

brdata = wload(datadir("sims", "brownian_data.jld2"))
Xd = brdata["Xd"]
ts = brdata["ts"]

Xd_rs = reshape(Xd, (size(Xd,1)*size(Xd,2), size(Xd,3)))
#Xd_rs = Xd_rs[:, 1:50]
Xd_rs = Xd_rs ./ sum(Xd_rs, dims=1)
# ========== define model ================= 

function build_br_sindyae(K::Int)
    
    enc = Chain(
        Dense(size(Xd_rs,1), 100, tanh),
        Dense(100, 10, tanh),
        Dense(10, K)
    )

    dec = Chain(
        Dense(K, 10, tanh),
        Dense(10, 100, tanh),
        Dense(100, size(Xd_rs,1)),
        Lux.WrappedFunction(softmax)
    )

    snet = SINDyNet([x -> PolyLib(x, order=5, use_cons=false)], [K,K])

    model = SINDyAE(enc, dec, snet)

    return model 
end

# ========== setup ================= 
rng = Random.default_rng()
Random.seed!(rng, 0)

opt = Optimisers.Adam(0.001f0)
vjp = AutoZygote()

ps, st = Lux.setup(rng, model)

tstate = Lux.Training.TrainState(model, ps, st, opt)

# ========== define model ================= 

rec_epochs = 0
dz_epochs = 5000
wind = 11
data = (Xd_rs,Xd_rs)
D, W = BuildWeightMat(ts, wind, 1), BuildWeightMat(ts, wind, 0)
tstate = train_sindyae(tstate, vjp, data, rec_epochs, dz_epochs, D, W; 
    mask_grads=true, lam1 = 0., lam2=1e0, lam3=1e-5, tau = 1e-1)

# evaluate 
enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
sindy_sf = StatefulLuxLayer{true}(tstate.model.sindy, tstate.parameters.sindy, tstate.states.sindy)

println(tstate.parameters.sindy.layer_1.weight)

p1 = heatmap(log10.(Xd_rs .+ 1e-4))
p2 = heatmap(log10.(dec_sf(enc_sf(Xd_rs)) .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(900,350)))

plot(enc_sf(Xd_rs)')