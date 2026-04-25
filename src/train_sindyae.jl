using LinearAlgebra, Setfield, Printf, Zygote, Lux, StatsBase

function sindae_loss(model, ps, st, data; lam1 = 0., lam2=1e-2, lam3=1e-4, D=nothing, W=nothing)
    x,dx = data
    enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
    dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
    snet_sf = StatefulLuxLayer{true}(model.sindy, ps.sindy, st.sindy)
  
    # ===== reconstruction error =======
    #z = enc_sf(x)
    #x_pred = dec_sf(z)
    #rec_loss = mean(abs2, x - x_pred)
    rec_loss = sum(abs2, x - dec_sf(enc_sf(x)))
    # ======= dxdt SINDy loss ========
    #dz_pred = snet_sf(z)
    #dx_pred = Lux.vector_jacobian_product(enc_sf, AutoZygote(), x, z)
    #dx_loss = mean(abs2, dx - dx_pred)
    #dx_loss = sum(abs2, dx - Lux.vector_jacobian_product(enc_sf, AutoZygote(), x, enc_sf(x)))
    #dx_loss = sum(abs2, D*x' - W*Lux.vector_jacobian_product(dec_sf, AutoZygote(), enc_sf(x), snet_sf(enc_sf(x)))')
    # ======= dzdt SINDy loss ========
    #dz = Lux.vector_jacobian_product(dec_sf, AutoZygote(), z, dx)
    #dz_loss = sum(abs2, Lux.vector_jacobian_product(dec_sf, AutoZygote(), enc_sf(x), dx) - snet_sf(enc_sf(x)))
    #dz_loss = sum(abs2, D*enc_sf(x)' - W*snet_sf(enc_sf(x))')
    @tensor dz[i,j,k] := D[i,l]*enc_sf(x)[j,l,k]
    @tensor dz_pred[i,j,k] := W[i,l]*snet_sf(enc_sf(x))[j,l,k]
    dz_loss = sum(abs2, dz - dz_pred)
    # ====== reg ===========
    reg_loss = sum(abs, ps.sindy.layer_1.weight)
    # ====== total loss ========
    #loss = rec_loss + lam1*dx_loss + lam2*dz_loss + lam3*reg_loss
    loss = rec_loss + lam2*dz_loss + lam3*reg_loss
  
    return loss, st, NamedTuple()
end

function train_sindyae(tstate, vjp, data, rec_epochs, dz_epochs, D, W; 
    mask_grads=true, lam1 = 0., lam2=1e1, lam3=1e-4, sparse_epoch=500, tau=1e-1)

    loss = Inf
    
    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end

    println("Reconstruction:")
    for epoch=1:rec_epochs
        # compute_gradients
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

        Lux.Training.apply_gradients!(tstate, grads)
        
        if epoch % 100 == 0 || epoch == rec_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    println("Updating remaining params:")
    for epoch=1:dz_epochs
        # compute_gradients
        loss_func(model, ps, st, data) = sindae_loss(model, ps, st, data, lam1=lam1, lam2=lam2, lam3=lam3, D=D, W=W)
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

        # get layers and mask

        if mask_grads
                
            grads_masked = get_grads_masked(grads, tstate)
            Lux.Training.apply_gradients!(tstate, grads_masked)

            if epoch % sparse_epoch == 0
                #tstate = pruned_tstate(tstate, tau)
                biginds = abs.(tstate.parameters.sindy.layer_1.weight) .> tau
                @set! tstate.model.sindy.layers.layer_1.Ξactive = biginds 
                @set! tstate.parameters.sindy.layer_1.weight = tstate.parameters.sindy.layer_1.weight .* biginds
            end
        else
            Lux.Training.apply_gradients!(tstate, grads)
        end

        if epoch % 100 == 0 || epoch == dz_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    return tstate
end  


function get_grads_masked(grads, tstate)
    #grads_masked = nothing
    args = keys(grads.sindy)

    sindy_grads = NamedTuple{args}((weight = grads.sindy[args[i]].weight .* tstate.model.sindy[i].Ξactive , bias = grads.sindy[args[i]].bias) for i=1:length(args))
    grads_masked = (enc = grads.enc, dec = grads.dec, sindy = sindy_grads)
    return grads_masked
end


function sweep_sindyae_models(rng, model, lams::Vector, taus::Vector, data, rec_epochs, dz_epochs, D, W; sparse_epoch=500)
    opt = Adam(0.001f0)
    vjp = AutoZygote()
    tstates = []
    params = []
    states = []
    combos = []
    scores = []
    for lam in lams 
        for tau in taus
            println("lam => $(lam), tau => $(tau)")
            push!(combos, Dict("lam" => lam, "tau" => tau))
            # train
            ps, st = Lux.setup(rng, model)
            tstate = Lux.Training.TrainState(model, ps, st, opt)
            tstate = train_sindyae(tstate, vjp, data, rec_epochs, dz_epochs, D, W; 
            mask_grads=true, lam1 = 0., lam2=1e1, lam3=lam, sparse_epoch=sparse_epoch, tau=tau)
            push!(tstates, tstate)
            push!(params, tstate.parameters)
            push!(states, tstate.states)
            # score
            k = count(abs.(tstate.parameters.sindy.layer_1.weight) .> 0) 
            model_z = StatefulLuxLayer{true}(tstate.model.sindy, tstate.parameters.sindy, tstate.states.sindy)
            enc_z = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
            #@tensor dz[i,j,k] := D[i,l]*enc_z(data[1])[j,l,k]
            #@tensor dz_pred[i,j,k] := W[i,l]*model_z(enc_z(data[1]))[j,l,k]
            mse = mean(abs2, D*enc_z(data[1])' - W*model_z(enc_z(data[1]))')
            #mse = sum(abs2, dz - dz_pred) / size(dz, 2)
            score = log(k*mse)
            push!(scores, score)
        end
    end
    return tstates, params, states, scores, combos
end

function sweep_sindyae_models(tstate_in::Lux.Training.TrainState, lams::Vector, taus::Vector, data, rec_epochs, dz_epochs, D, W; sparse_epoch=500)
    vjp = AutoZygote()
    tstates = []
    params = []
    states = []
    combos = []
    scores = []
    for lam in lams 
        for tau in taus
            println("lam => $(lam), tau => $(tau)")
            push!(combos, Dict("lam" => lam, "tau" => tau))
            # train
            tstate = deepcopy(tstate_in)
            tstate = train_sindyae(tstate, vjp, data, rec_epochs, dz_epochs, D, W; 
            mask_grads=true, lam1 = 0., lam2=1e1, lam3=lam, sparse_epoch=sparse_epoch, tau=tau)
            push!(tstates, tstate)
            push!(params, tstate.parameters)
            push!(states, tstate.states)
            # score
            k = count(abs.(tstate.parameters.sindy.layer_1.weight) .> 0) 
            model_z = StatefulLuxLayer{true}(tstate.model.sindy, tstate.parameters.sindy, tstate.states.sindy)
            enc_z = StatefulLuxLayer{true}(tstate.model.enc, tstate.parameters.enc, tstate.states.enc)
            #@tensor dz[i,j,k] := D[i,l]*enc_z(data[1])[j,l,k]
            #@tensor dz_pred[i,j,k] := W[i,l]*model_z(enc_z(data[1]))[j,l,k]
            mse = mean(abs2, D*enc_z(data[1])' - W*model_z(enc_z(data[1]))')
            #mse = sum(abs2, dz - dz_pred) / size(dz, 2)
            score = log(k*mse)
            push!(scores, score)
        end
    end
    return tstates, params, states, scores, combos
end

function update_sindy_dynamics_only(
    tstate::Lux.Training.TrainState,
    vjp,
    data,
    rec_epochs,
    dz_epochs,
    D,
    W;
    lam1 = 0.,
    lam2 = 1e1,
    lam3 = 1e-4,
    tau = 1e-1,
    sparse_epoch = 500,
    mask_grads = true
    )

    println("Adapting dynamics only (encoder/decoder frozen):")

    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end
    
    zero_like(x) = zero(x)           # arrays, numbers
    zero_like(::Nothing) = nothing   # softmax / param-free layers

    function zero_tree(nt::NamedTuple)
        NamedTuple{keys(nt)}(zero_tree.(values(nt)))
    end

    zero_tree(x) = zero_like(x)
    
    println("Reconstruction:")
    for epoch=1:rec_epochs
        # compute_gradients
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            enc = zero_tree(grads.enc),
            #enc = grads.enc,
            dec   = zero_tree(grads.dec),
            #dec = grads.dec,
            sindy = Lux.fmap(x-> zero(x), tstate.parameters.sindy)
        )
        # --------------------------
        # Optional sparsity masking 
        # --------------------------
        
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end
         
        Lux.Training.apply_gradients!(tstate, grads_frozen)
        
        if epoch % 100 == 0 || epoch == rec_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    println("dzdt:")
    for epoch in 1:dz_epochs

        loss_func(model, ps, st, data) =
            sindae_loss(model, ps, st, data,
                        lam1 = lam1, lam2 = lam2, lam3 = lam3,
                        D = D, W = W)

        grads, loss, _, _ =
            Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

        # -------------------------------------------------
        # ❄️ FREEZE encoder & decoder
        # -------------------------------------------------

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            enc   = zero_tree(grads.enc),
            #enc = grads.enc,
            dec   = zero_tree(grads.dec),
            #dec = grads.dec,
            sindy = grads.sindy
        )

        # -------------------------------------------------
        # Optional sparsity masking (your existing logic)
        # -------------------------------------------------
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end

        Lux.Training.apply_gradients!(tstate, grads_frozen)

        # -------------------------------------------------
        # Hard threshold pruning (unchanged)
        # -------------------------------------------------
        if mask_grads && epoch % sparse_epoch == 0
            biginds = abs.(tstate.parameters.sindy.layer_1.weight) .> tau

            @set! tstate.model.sindy.layers.layer_1.Ξactive = biginds
            @set! tstate.parameters.sindy.layer_1.weight =
                tstate.parameters.sindy.layer_1.weight .* biginds
        end

        if epoch % 100 == 0 || epoch == dz_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    return tstate
end

function update_sindy_dynamics_only(
    tstate::Lux.Training.TrainState,
    vjp,
    data,
    dz_epochs,
    D,
    W;
    lam1 = 0.,
    lam2 = 1e1,
    lam3 = 1e-4,
    tau = 1e-1,
    sparse_epoch = 500,
    mask_grads = true
    )

    println("Adapting dynamics only (encoder/decoder frozen):")

    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end
    
    zero_like(x) = zero(x)           # arrays, numbers
    zero_like(::Nothing) = nothing   # softmax / param-free layers

    function zero_tree(nt::NamedTuple)
        NamedTuple{keys(nt)}(zero_tree.(values(nt)))
    end

    zero_tree(x) = zero_like(x)

    println("dzdt:")
    for epoch in 1:dz_epochs

        loss_func(model, ps, st, data) =
            sindae_loss(model, ps, st, data,
                        lam1 = lam1, lam2 = lam2, lam3 = lam3,
                        D = D, W = W)

        grads, loss, _, _ =
            Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

        # -------------------------------------------------
        # ❄️ FREEZE encoder & decoder
        # -------------------------------------------------

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            enc   = zero_tree(grads.enc),
            #enc = grads.enc
            dec   = zero_tree(grads.dec),
            #dec = grads.dec,
            sindy = grads.sindy
        )

        # -------------------------------------------------
        # Optional sparsity masking (your existing logic)
        # -------------------------------------------------
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end

        Lux.Training.apply_gradients!(tstate, grads_frozen)

        # -------------------------------------------------
        # Hard threshold pruning (unchanged)
        # -------------------------------------------------
        if mask_grads && epoch % sparse_epoch == 0
            biginds = abs.(tstate.parameters.sindy.layer_1.weight) .> tau

            @set! tstate.model.sindy.layers.layer_1.Ξactive = biginds
            @set! tstate.parameters.sindy.layer_1.weight =
                tstate.parameters.sindy.layer_1.weight .* biginds
        end

        if epoch % 100 == 0 || epoch == dz_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    return tstate
end

function update_sindy_dynamics_and_enc(
    tstate::Lux.Training.TrainState,
    vjp,
    data,
    rec_epochs,
    dz_epochs,
    D,
    W;
    lam1 = 0.,
    lam2 = 1e1,
    lam3 = 1e-4,
    tau = 1e-1,
    sparse_epoch = 500,
    mask_grads = true
    )

    println("Adapting dynamics only (encoder/decoder frozen):")

    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end
    
    zero_like(x) = zero(x)           # arrays, numbers
    zero_like(::Nothing) = nothing   # softmax / param-free layers

    function zero_tree(nt::NamedTuple)
        NamedTuple{keys(nt)}(zero_tree.(values(nt)))
    end

    zero_tree(x) = zero_like(x)
    
    println("Reconstruction:")
    for epoch=1:rec_epochs
        # compute_gradients
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            #enc = zero_tree(grads.enc),
            enc = grads.enc,
            dec   = zero_tree(grads.dec),
            #dec = grads.dec,
            sindy = Lux.fmap(x-> zero(x), tstate.parameters.sindy)
        )
        # --------------------------
        # Optional sparsity masking 
        # --------------------------
        
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end
         
        Lux.Training.apply_gradients!(tstate, grads_frozen)
        
        if epoch % 100 == 0 || epoch == rec_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    println("dzdt:")
    for epoch in 1:dz_epochs

        loss_func(model, ps, st, data) =
            sindae_loss(model, ps, st, data,
                        lam1 = lam1, lam2 = lam2, lam3 = lam3,
                        D = D, W = W)

        grads, loss, _, _ =
            Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

        # -------------------------------------------------
        # ❄️ FREEZE encoder & decoder
        # -------------------------------------------------

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            #enc   = zero_tree(grads.enc),
            enc = grads.enc,
            dec   = zero_tree(grads.dec),
            #dec = grads.dec,
            sindy = grads.sindy
        )

        # -------------------------------------------------
        # Optional sparsity masking (your existing logic)
        # -------------------------------------------------
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end

        Lux.Training.apply_gradients!(tstate, grads_frozen)

        # -------------------------------------------------
        # Hard threshold pruning (unchanged)
        # -------------------------------------------------
        if mask_grads && epoch % sparse_epoch == 0
            biginds = abs.(tstate.parameters.sindy.layer_1.weight) .> tau

            @set! tstate.model.sindy.layers.layer_1.Ξactive = biginds
            @set! tstate.parameters.sindy.layer_1.weight =
                tstate.parameters.sindy.layer_1.weight .* biginds
        end

        if epoch % 100 == 0 || epoch == dz_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    return tstate
end

function update_sindy_dynamics_and_dec(
    tstate::Lux.Training.TrainState,
    vjp,
    data,
    rec_epochs,
    dz_epochs,
    D,
    W;
    lam1 = 0.,
    lam2 = 1e1,
    lam3 = 1e-4,
    tau = 1e-1,
    sparse_epoch = 500,
    mask_grads = true
    )

    println("Adapting dynamics only (encoder/decoder frozen):")

    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end
    
    zero_like(x) = zero(x)           # arrays, numbers
    zero_like(::Nothing) = nothing   # softmax / param-free layers

    function zero_tree(nt::NamedTuple)
        NamedTuple{keys(nt)}(zero_tree.(values(nt)))
    end

    zero_tree(x) = zero_like(x)
    
    println("Reconstruction:")
    for epoch=1:rec_epochs
        # compute_gradients
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            enc = zero_tree(grads.enc),
            #enc = grads.enc,
            #dec   = zero_tree(grads.dec),
            dec = grads.dec,
            sindy = Lux.fmap(x-> zero(x), tstate.parameters.sindy)
        )
        # --------------------------
        # Optional sparsity masking 
        # --------------------------
        
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end
         
        Lux.Training.apply_gradients!(tstate, grads_frozen)
        
        if epoch % 100 == 0 || epoch == rec_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    println("dzdt:")
    for epoch in 1:dz_epochs

        loss_func(model, ps, st, data) =
            sindae_loss(model, ps, st, data,
                        lam1 = lam1, lam2 = lam2, lam3 = lam3,
                        D = D, W = W)

        grads, loss, _, _ =
            Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

        # -------------------------------------------------
        # ❄️ FREEZE encoder & decoder
        # -------------------------------------------------

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            enc   = zero_tree(grads.enc),
            #enc = grads.enc,
            #dec   = zero_tree(grads.dec),
            dec = grads.dec,
            sindy = grads.sindy
        )

        # -------------------------------------------------
        # Optional sparsity masking (your existing logic)
        # -------------------------------------------------
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end

        Lux.Training.apply_gradients!(tstate, grads_frozen)

        # -------------------------------------------------
        # Hard threshold pruning (unchanged)
        # -------------------------------------------------
        if mask_grads && epoch % sparse_epoch == 0
            biginds = abs.(tstate.parameters.sindy.layer_1.weight) .> tau

            @set! tstate.model.sindy.layers.layer_1.Ξactive = biginds
            @set! tstate.parameters.sindy.layer_1.weight =
                tstate.parameters.sindy.layer_1.weight .* biginds
        end

        if epoch % 100 == 0 || epoch == dz_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    return tstate
end

function update_sindy_all(
    tstate::Lux.Training.TrainState,
    vjp,
    data,
    rec_epochs,
    dz_epochs,
    D,
    W;
    lam1 = 0.,
    lam2 = 1e1,
    lam3 = 1e-4,
    tau = 1e-1,
    sparse_epoch = 500,
    mask_grads = true
    )

    println("Adapting dynamics only (encoder/decoder frozen):")

    function rec_loss(model, ps, st, data)
        x,y = data
        enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
        dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
        loss = sum(abs2, x - dec_sf(enc_sf(x)))
        return loss, st, NamedTuple()
    end
    
    zero_like(x) = zero(x)           # arrays, numbers
    zero_like(::Nothing) = nothing   # softmax / param-free layers

    function zero_tree(nt::NamedTuple)
        NamedTuple{keys(nt)}(zero_tree.(values(nt)))
    end

    zero_tree(x) = zero_like(x)
    
    println("Reconstruction:")
    for epoch=1:rec_epochs
        # compute_gradients
        grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            #enc = zero_tree(grads.enc),
            enc = grads.enc,
            #dec   = zero_tree(grads.dec),
            dec = grads.dec,
            sindy = Lux.fmap(x-> zero(x), tstate.parameters.sindy)
        )
        # --------------------------
        # Optional sparsity masking 
        # --------------------------
        
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end
         
        Lux.Training.apply_gradients!(tstate, grads_frozen)
        
        if epoch % 100 == 0 || epoch == rec_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    println("dzdt:")
    for epoch in 1:dz_epochs

        loss_func(model, ps, st, data) =
            sindae_loss(model, ps, st, data,
                        lam1 = lam1, lam2 = lam2, lam3 = lam3,
                        D = D, W = W)

        grads, loss, _, _ =
            Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

        # -------------------------------------------------
        # ❄️ FREEZE encoder & decoder
        # -------------------------------------------------

        # ❄️ Freeze encoder & decoder
        grads_frozen = (
            #enc   = zero_tree(grads.enc),
            enc = grads.enc,
            #dec   = zero_tree(grads.dec),
            dec = grads.dec,
            sindy = grads.sindy
        )

        # -------------------------------------------------
        # Optional sparsity masking (your existing logic)
        # -------------------------------------------------
        if mask_grads
            grads_frozen = get_grads_masked(grads_frozen, tstate)
        end

        Lux.Training.apply_gradients!(tstate, grads_frozen)

        # -------------------------------------------------
        # Hard threshold pruning (unchanged)
        # -------------------------------------------------
        if mask_grads && epoch % sparse_epoch == 0
            biginds = abs.(tstate.parameters.sindy.layer_1.weight) .> tau

            @set! tstate.model.sindy.layers.layer_1.Ξactive = biginds
            @set! tstate.parameters.sindy.layer_1.weight =
                tstate.parameters.sindy.layer_1.weight .* biginds
        end

        if epoch % 100 == 0 || epoch == dz_epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end

    return tstate
end