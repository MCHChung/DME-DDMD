using LuxCore, WeightInitializers, Random, Optimisers, Printf, StatsBase

struct KoopmanAE{L1, L2, L3} <: LuxCore.AbstractLuxContainerLayer{(:enc, :dec, :K)}
    enc::L1
    dec::L2
    K::L3 # linear
  end

function koopmanae_loss(model, ps, st, data; a1= 1e-1, a2=1e-7, a3=1e-14, Sp=30)
    x , y = data
    enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
    dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
    K_sf = StatefulLuxLayer{true}(model.K, ps.K, st.K)
    
    # reconstruction
    rec_loss = sum(abs2, x - dec_sf(enc_sf(x)))
  
    # proj fwd and reconstruct
    fwd_recon_loss = 0.
    xhat = enc_sf(x)
    for i=1:Sp
      xhat = K_sf(xhat)
      #fwd_recon_loss += mean(abs2, x[:, i+1:end] - dec_sf(Km(enc_sf(x[:,1:end-i]), i)))
      fwd_recon_loss += sum(abs2, x[:, i+1:end, :] - dec_sf(xhat[:,1:end-i, :]))
    end
    fwd_recon_loss = fwd_recon_loss/Sp
  
    rec_loss = rec_loss + fwd_recon_loss
  
    # linear loss
    lin_loss = 0.
    T = size(x,2)
    T = Sp
    xhat = enc_sf(x)
    for i=1:(T-1)
      xhat = K_sf(xhat)
      #lin_loss += mean(abs2, enc_sf(x[:, i+1:end]) - Km(enc_sf(x[:,1:end-i]), i))
      lin_loss += sum(abs2, enc_sf(x[:, i+1:end, :]) - xhat[:,1:end-i, :])
    end
    lin_loss = lin_loss/(T-1)
    
    # max loss
    rec_max_loss = sum(x -> x^6, x - dec_sf(enc_sf(x)))
    fwd_max_loss = sum(x -> x^6, x[:, 2:end] - dec_sf(K_sf(enc_sf(x[:, 1:end-1]))))
    max_loss = rec_max_loss + fwd_max_loss
    
    # reg loss
    reg_loss = sum(abs2, first(Optimisers.destructure(ps)))
  
    loss = a1*rec_loss + lin_loss + a2*max_loss + a3*reg_loss
    #loss = a1*rec_loss + lin_loss + a3*reg_loss
  
    return loss, st, NamedTuple()
end  


function train_koopmanae(tstate, vjp, data, rec_epochs, K_epochs; 
    iter=Int(floor(epochs/10)), a1= 1e-1, a2=1e-7, a3=1e-14, Sp=30)

    loss = Inf

    function rec_loss(model, ps,st, data)
      x , y = data
      enc_sf = StatefulLuxLayer{true}(model.enc, ps.enc, st.enc)
      dec_sf = StatefulLuxLayer{true}(model.dec, ps.dec, st.dec)
      loss = sum(abs2, x - dec_sf(enc_sf(x)))
      return loss, st, NamedTuple()
    end

    println("Reconstruction:")
    for epoch in 1:rec_epochs
      # compute_gradients
      grads, loss, _, _ = Lux.Training.compute_gradients(vjp, rec_loss, data, tstate)

      # get layers and mask
      Lux.Training.apply_gradients!(tstate, grads)

      if epoch % iter == 0 || epoch == epochs
          @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
      end
  end

  println("K:")
  for epoch in 1:K_epochs
      # compute_gradients
      loss_func(model,ps,st,data) = koopmanae_loss(model, ps, st, data; a1= a1, a2=a2, a3=a3, Sp=Sp)
      grads, loss, _, _ = Lux.Training.compute_gradients(vjp, loss_func, data, tstate)

      # get layers and mask
      Lux.Training.apply_gradients!(tstate, grads)

      if epoch % iter == 0 || epoch == epochs
          @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
      end
  end

    return tstate
end