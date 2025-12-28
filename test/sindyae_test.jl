using Plots, DrWatson, Lux , Optimisers, Zygote

@quickactivate "TMI&MD"

include(srcdir("build_weight_mats.jl"))
include(srcdir("train_sindyae.jl"))
include(srcdir("sindyae.jl"))
include(srcdir("libs.jl"))

# ========== gen data ================= 
xs = -1.5:0.01:1.5
ts = 0.:0.01:1.
p = zeros((length(xs), length(ts)))

for t=1:length(ts)
  pt = @. exp(-100*xs^2/(t + 1e-2))
  pt = pt ./ sum(pt)
  p[:,t] = pt
end

# ========== define model ================= 
enc = Chain(
  Dense(size(p,1), 100, tanh),
  Dense(100, 10, tanh),
  Dense(10, 1)
)

dec = Chain(
  Dense(1, 10, tanh),
  Dense(10, 100, tanh),
  Dense(100, size(p,1)),
  Lux.WrappedFunction(softmax)
)

snet = SINDyNet([x -> PolyLib(x, order=5, use_cons=false)], [1,1])

model = SINDyAE(enc, dec, snet)

# ========== setup ================= 
rng = Random.default_rng()
Random.seed!(rng, 0)

opt = Optimisers.Adam(0.001f0)
vjp = AutoZygote()

ps, st = Lux.setup(rng, model)

tstate = Lux.Training.TrainState(model, ps, st, opt)

# ========== define model ================= 

rec_epochs = 0
dz_epochs = 7000
wind = 11
data = (p,p)
D, W = BuildWeightMat(ts, wind, 1), BuildWeightMat(ts, wind, 0)
tstate = train_sindyae(tstate, vjp, data, rec_epochs, dz_epochs, D, W; 
    mask_grads=true, lam1 = 0., lam2=1e0, lam3=1e-5, tau = 1e-4)

# evaluate 
enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
sindy_sf = StatefulLuxLayer{true}(tstate.model.sindy, tstate.parameters.sindy, tstate.states.sindy)

println(tstate.parameters.sindy.layer_1.weight)

p1 = heatmap(log10.(p .+ 1e-4))
p2 = heatmap(log10.(dec_sf(enc_sf(p)) .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(900,350)))

plot(enc_sf(p)')