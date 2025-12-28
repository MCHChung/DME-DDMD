using Plots, DrWatson, Lux , Optimisers, Zygote

@quickactivate "TMI&MD"

include(srcdir("KoopmanAE.jl"))

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
Klat = 3
enc = Chain(
  Dense(size(p,1), 100, tanh),
  Dense(100, 10, tanh),
  Dense(10, Klat)
)

dec = Chain(
  Dense(Klat, 10, tanh),
  Dense(10, 100, tanh),
  Dense(100, size(p,1)),
  Lux.WrappedFunction(softmax)
)

K = Chain(Dense(Klat => Klat, use_bias=false))

model = KoopmanAE(enc, dec, K)

# ========== setup ================= 
rng = Random.default_rng()
Random.seed!(rng, 0)

opt = Optimisers.Adam(0.001f0)
vjp = AutoZygote()

ps, st = Lux.setup(rng, model)

tstate = Lux.Training.TrainState(model, ps, st, opt)

# ========== define model ================= 

data = (p,p)
rec_epochs = 0
K_epochs = 5000
tstate = train_koopmanae(tstate, vjp, data, rec_epochs, K_epochs; 
  iter=100, a1= 1e1, a2=1e-7, a3=1e-14, Sp=20)

# evaluate 
enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
K_sf = StatefulLuxLayer{true}(tstate.model.K, tstate.parameters.K, tstate.states.K)

println(tstate.parameters.K.layer_1.weight)

p1 = heatmap(log10.(p .+ 1e-4))
p2 = heatmap(log10.(dec_sf(enc_sf(p)) .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(900,350)))

perr = heatmap(p .* log.((p .+ 1e-4) ./ (dec_sf(enc_sf(p)) .+ 1e-4)))
#perr = heatmap(p .* log.((p .+ 1e-4) ./ (dec_sf(enc_sf(p)) .+ 1e-4)))
display(perr)

display(plot(enc_sf(p)'))