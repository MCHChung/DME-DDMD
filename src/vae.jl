using LuxCore, WeightInitializers, Random, Optimisers, Printf, StatsBase

struct VAE{E,D} <: LuxCore.AbstractLuxContainerLayer{(:enc, :dec)}
    enc::E
    dec::D
end

function Encoder(D, K, hidden=64)
    return Chain(
        Dense(D, hidden, relu),
        Dense(hidden, hidden, relu),
        Dense(hidden, 2K)   # outputs [μ_z; logσ_z²]
    )
end

function Decoder(K, D, hidden=64)
    return Chain(
        Dense(K, hidden, relu),
        Dense(hidden, hidden, relu),
        Dense(hidden, D)   # mean of p(x|z)
    )
end

function reparameterize(μ, logσ2, rng)
    ϵ = randn(rng, size(μ))
    return μ .+ exp.(0.5 .* logσ2) .* ϵ
end

function forward(model::VAE, x, ps, st, rng)
    enc = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
    dec = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)

    h = enc(x)
    K = size(h, 1) ÷ 2

    μz = h[1:K, :]
    logσ2z = h[K+1:end, :]

    z = reparameterize(μz, logσ2z, rng)
    μx = dec(z)

    return μx, μz, logσ2z
end

function vae_loss(model, ps, st, x; σx2=1.0, rng=Random.default_rng())
    μx, μz, logσ2z = forward(model, x, ps, st, rng)

    # Reconstruction loss: −E_q log p(x|z)
    recon = sum((x .- μx).^2) / (2σx2)

    # KL divergence: KL(q(z|x) || p(z))
    kl = 0.5 * sum(exp.(logσ2z) .+ μz.^2 .- 1 .- logσ2z)

    return recon + kl
end


function train_vae(vae)
    for epoch in 1:200
        loss, back = Zygote.pullback(ps) do p
            vae_loss(vae, p, st, x_data; rng=rng)
        end
        grads = back(1.0)[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grads)
    
        if epoch % 20 == 0
            @info "Epoch $epoch | Loss = $loss"
        end
    end
end