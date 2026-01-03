using Plots, DrWatson, JLD2 , Lux, LuxCore, Optimisers, Zygote, Printf, Random

@quickactivate "TMI&MD"

include(srcdir("nnmf.jl"))

struct AE{L1, L2} <: LuxCore.AbstractLuxContainerLayer{(:enc, :dec)}
    enc::L1
    dec::L2
end 

function ae_model(sys::Int, in_dim, K::Int)
    
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
        return AE(enc, dec)
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
        return AE(enc, dec)
    end
     
    #return AE(enc, dec)
end

function train_ae(tstate, vjp, data, epochs)

    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end

    loss = Inf
    for epoch=1:epochs
        # compute_gradients
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

        Lux.Training.apply_gradients!(tstate, grads)
        
        if epoch % 100 == 0 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    println("Updating remaining params:")
    
    return tstate
end 

function kld_loss(p, tstate::Lux.Training.TrainState; eps=1e-8)
    enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
    dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
    q = dec_sf(enc_sf(p))
    return kld_loss(p,q,eps=eps)
end

function sweep_ae_models(rng, sys::Int, X, Ks; epochs=5000)
    klds_ae = zeros(length(Ks))
    vjp = AutoZygote()
    opt = Adam(0.001f0)
    for i=1:length(Ks)
        println("K => $(Ks[i])")
        model = ae_model(sys, size(X,1), Ks[i])
        ps,st = Lux.setup(rng,model)
        tstate = Lux.Training.TrainState(model, ps ,st, opt)
        tstate = train_ae(tstate, vjp, (X,X), epochs)
        kld = kld_loss(X, tstate)
        klds_ae[i] = kld
    end

    return klds_ae
end

br_data = wload(datadir("sims", "brownian_data.jld2"))
ou_data = wload(datadir("sims", "ou_data.jld2"))
np_data = wload(datadir("sims", "np_data.jld2"))
neuron_data = wload(datadir("sims", "neuron_data.jld2"))

Ks = collect(1:20)
epochs = 10000
rng = Random.default_rng()
Random.seed!(rng, 0)

# load data
# ======= brownian ========
Xbr = br_data["Xd"]
Xbr = reshape(Xbr, (size(Xbr,1)*size(Xbr,2), size(Xbr,3)))
Xbr = Xbr ./ sum(Xbr, dims=1)
# ======= OU =========
Xou = ou_data["Xd"]
Xou = Xou ./ sum(Xou, dims=1)
# ======= NP =========
Xnp = np_data["Ns"]
Xnp = Xnp ./ sum(Xnp, dims=1)
# ======= neuron =========
P = neuron_data["Psms_tr"]
Xne1, Xne2 = abs.(P[1]), abs.(P[2])
Xne1 = Xne1 ./ sum(Xne1, dims=1)
Xne2 = Xne2 ./ sum(Xne2, dims=1)

# sweep models
println("Brownian")
klds_br_ae = sweep_ae_models(rng, 1, Xbr, Ks; epochs=epochs)
println("OU")
klds_ou_ae = sweep_ae_models(rng, 2, Xou, Ks; epochs=epochs)
println("NP")
klds_np_ae = sweep_ae_models(rng, 3, Xnp, Ks; epochs=epochs)
println("Neuron 1")
klds_ne1_ae = sweep_ae_models(rng, 4, Xne1, Ks; epochs=epochs)
println("Neuron 2")
klds_ne2_ae = sweep_ae_models(rng, 4 , Xne2, Ks; epochs=epochs)

data_ae_dict = @strdict klds_br_ae klds_ou_ae klds_np_ae klds_ne1_ae klds_ne2_ae


# =========== Brownian =============
K = 1
sys = 1
epochs = 5000
model = ae_model(sys, size(Xbr,1), K)
ps,st = Lux.setup(rng,model)
vjp = AutoZygote()
opt = Adam(0.001f0)
tstate = Lux.Training.TrainState(model, ps ,st, opt)
tstate = train_ae(tstate, vjp, (Xbr,Xbr), epochs)

enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
Xbr_pred = dec_sf(enc_sf(Xbr))

p1 = heatmap(log10.(Xbr[:,:,1] .+ 1e-4))
p2 = heatmap(log10.(Xbr_pred[:,:,1] .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(850,300)))

ind = 500
p1 = heatmap(reshape(Xbr[:, ind], (41,41))); 
p2 = heatmap(reshape(Xbr_pred[:, ind], (41,41)))
display(plot(p1,p2,layout=(1,2), size=(850,300)))

# =========== OU =============
K = 1
sys = 2
epochs = 5000
model = ae_model(sys, size(Xou,1), K)
ps,st = Lux.setup(rng, model)
vjp = AutoZygote()
opt = Adam(0.001f0)
tstate = Lux.Training.TrainState(model, ps ,st, opt)
tstate = train_ae(tstate, vjp, (Xou,Xou), epochs)

enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
Xou_pred = dec_sf(enc_sf(Xou))

p1 = heatmap(log10.(Xou[:,:,1] .+ 1e-4))
p2 = heatmap(log10.(Xou_pred[:,:,1] .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(850,300)))

ind = 50
plot(Xou[:, ind]); plot!(Xou_pred[:, ind])

# =========== NP =============
K = 1
epochs = 10000
model = ae_model(size(Xnp,1), K)
ps,st = Lux.setup(rng,model)
vjp = AutoZygote()
opt = Adam(0.001f0)
tstate = Lux.Training.TrainState(model, ps ,st, opt)
tstate = train_ae(tstate, vjp, (Xnp,Xnp), epochs)

enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
Xnp_pred = dec_sf(enc_sf(Xnp))

p1 = heatmap(log10.(Xnp[:,:,1] .+ 1e-4))
p2 = heatmap(log10.(Xnp_pred[:,:,1] .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(850,300)))

ind = 60
plot(Xnp[ind, :]); plot!(Xnp_pred[ind, :])

# ======== Neuron 1 test ===========
K = 2
model = ae_model(size(Xne1,1), K)
ps,st = Lux.setup(rng,model)
vjp = AutoZygote()
opt = Adam(0.001f0)
tstate = Lux.Training.TrainState(model, ps ,st, opt)
tstate = train_ae(tstate, vjp, (Xne1,Xne1), epochs)

enc_sf = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
dec_sf = StatefulLuxLayer{true}(tstate.model.dec, tstate.parameters.dec, tstate.states.dec)
Xne1_pred = dec_sf(enc_sf(Xne1))

p1 = heatmap(log10.(Xne1[:,:,1] .+ 1e-4))
p2 = heatmap(log10.(Xne1_pred[:,:,1] .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(850,300)))





# =============== svd ==================
K = 40
U,S,V = svd(log.(Xou .+ 1e-8))

A = U[:, 1:K] * diagm(S[1:K]) * V[:, 1:K]' 

Xou_svd = exp.(A)
Xou_svd = Xou_svd ./ sum(Xou_svd, dims=1)

p1 = heatmap(log10.(Xou .+ 1e-4))
p2 = heatmap(log10.(Xou_svd .+ 1e-4))
display(plot(p1,p2,layout=(1,2), size=(850,300)))

print(kld_loss(Xou, Xou_svd))
# ======================================