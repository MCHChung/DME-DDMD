using LinearAlgebra ,  StatsBase, TensorOperations, Random

# ========================== SPARSE REGRESSION ===============================
# INPUTS (for AdSR)
# Θ: The library matrix with size n x p, where n is the data length, p is the number of nonlinear basis. 
# y: Output, for dynamics will be estimated or measured derivative of dynamics.
# iter: Number of regressions you would like to perform.
# c : adds optional additional sparsity enforcement 

# OUTPUTS (for AdSR)
# Ξes = estimated model 
# min_score = min IC score

function AdSR(θ, y, ic::String; iter = 10, c=0., trainpct=80, abstol=1e-7, reltol = 1e-7, ρ=0, Ξactive=nothing, use_cond=true)
    @assert ic == "slic" || ic == "aicc" || ic == "bic"

    # random train-val split 
    nobs , n_state = size(y)
    bag_size = Int(floor(trainpct*nobs/100))
    traininds = sort(sample(1:nobs, bag_size, replace=false))
    testinds = [i for i=1:nobs if i ∉ traininds]
    θ_train = θ[traininds, :] ; θ_test = θ[testinds, :]
    y_train = y[traininds, :] ; y_test = y[testinds, :]
    # define information criterion scoring procedure 
    η = use_cond ? c*cond(θ_train) : c # this is optional enforcement to prevent log(RSS) -> ∞ , SLIC does well without it!
    function score(Ξ, ic)
        nobs_test = size(y_test, 1) # number of observations in test/val 
        k = count(abs.(Ξ) .> 0.) + 1 # number of free params
        RSS = sum(abs2, y_test - θ_test*Ξ)
        if ic=="slic"
            nobs_test*log(k*(RSS + η)/nobs_test)
        elseif ic=="aicc"
            nobs_test*log((RSS + η)/nobs_test) + 2*k*nobs_test/(nobs_test-k-1)
        elseif ic=="bic"
            nobs_test*log((RSS + η)/nobs_test) + k*log(nobs_test)
        else
            throw("Please enter valid selection criterion => slic, aicc, bic")
        end
    end
    # Initialize comparison values
    min_score = Inf
    prev_smallinds = [1]

    # Get an initial estimate of the selection matrix Ξes
    Ξes = θ_train \ y_train    
    Ξactive = Ξactive == nothing ? ones(Int, size(Ξes)) : Ξactive
    Ξes = update_active_coefs!(θ_train, y_train, Ξes, Ξactive)
    X_prev = θ_train * Ξes 

    for i=1:iter

        # auto-gen thresholds 
        λmin = min(abs.(Ξes[Ξes .!= 0])...) 
        λmax = max(abs.(Ξes[Ξes .!= 0])...)
        λs = range(λmin, λmax, abs(1000*Int(ceil(log10(λmax/λmin)))))
        
        # sparsify and score effect 
        for λ in λs
            # Make a temporary Ξes matrix to test out the effect of λ
            temp_Ξes = copy(Ξes)
            # Get the index of values whose absolute value is smaller than λ
            smallinds = (abs.(temp_Ξes).<λ)

            # If the effect of λ is the same as the previous one, no need to do calculations again
            if smallinds == prev_smallinds
                continue
            end

            # Set the parameter value of library term whose absolute value is smaller than λ as zero
            temp_Ξes[smallinds] .= 0
            #any(all(abs.(temp_Ξes) .== 0, dims=1)) ? break : nothing
            # Regress the dynamics to the remaining terms
            for ind=1:n_state
                biginds = .!smallinds[:,ind]
                #temp_Ξes[biginds,ind] = θ_train[:,biginds]\y_train[:,ind]
                temp_Ξes[biginds,ind] = inv(θ_train[:,biginds]'θ_train[:,biginds] + ρ*I)*θ_train[:,biginds]'y_train[:,ind]
            end
            any(all(abs.(temp_Ξes) .== 0, dims=1)) ? break : nothing
            # Save the current small indices
            prev_smallinds = smallinds

            # calculate the loss and compare it to our best loss
            score_iter = score(temp_Ξes, ic)

            #loss = DDMDLoss2(du,θ,temp_Ξes) 
            if score_iter < min_score
                Ξes = copy(temp_Ξes)
                min_score = score_iter
            end
        end
        
        X = θ_train * Ξes # make new SINDy prediction
        
        # If nothing, or very little, changed in one iteration, then we have converged
        if _is_converged(X, X_prev, abstol, reltol)
            break
        end
        
        X_prev = X
    end

    return Ξes, min_score
end

function _is_converged(X, X_prev, abstol, reltol)::Bool
    Δ = norm(X .- X_prev)
    Δ < abstol && return true
    δ = Δ / norm(X)
    δ < reltol && return true
    return false
end

# This does AdSR with ensembling. The final output, ips, is the matrix of inclusion probabilities
function EnAdSR(θ, y, ic::String;
        c = 0.,
        trainpct = 80,
        num_batches = 10,
        iter=10,
        tol = 0.7, 
        ρ = 0, 
        Ξactive = nothing,
        use_cond=true
    )
    # hold ensemble of models  
    ΞB = zeros((size(θ\y)..., num_batches))
    scores = zeros(num_batches)
    # determine number of sample size
    N = size(y,1)

    for i=1:num_batches
        # get model and score for this random train-val split 
        Ξes , score = AdSR(θ, y, ic; iter=iter, c=c, trainpct=trainpct, ρ=ρ, Ξactive=Ξactive, use_cond=use_cond)
        # add to holders 
        ΞB[:,:,i] = Ξes
        scores[i] = score
    end

    # compute the inclusion probabilities for model coefficients 
    biginds = abs.(ΞB).>0
    ips = mean(biginds , dims=3)

    # Compute the ensembled Ξs and probabilistically prune
    Ξes = sum(ΞB, dims=3)./count(biginds,dims=3) 
    Ξes[ips .< tol] .= 0 
    Ξes = Ξes[:,:,1]

    # final regression to update
    n_state = size(y, 2)
    smallinds = .!(abs.(Ξes) .> 0)
    for ind=1:n_state
        biginds = .!smallinds[:,ind]
        Ξes[biginds,ind] = θ[:,biginds]\y[:,ind]
    end

    return Ξes, scores, ips[:,:,1]
end


# ================================= TMI ========================================
# INPUTS 
# X: data 
# K: latent space dimensionality
# W: Galerkin matrix for states
# DW : Galerkin matrix for derivatives of states 
# N: Polynomial order for library for dzdt 
# θy: library of model terms for Y   

# OUTPUTS 
# (z , Y, q): (time-dep latents, feature-dep latents, compressed data matrix)
# (Ξz, Ξy): (model for z, model for Y)
# errs: 

function tmi(X::Array{Float64, 2}, K::Int; 
    normalize=true,
    maxiters=100000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=false, 
    E=nothing, 
    z=nothing, 
    Y=nothing, 
    νy = 0,
    νz = 0, 
    track_lats = false,
    ρ = 0 , 
    ptol = 0.7,
    print_err = true,
    seed=0
    )

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # get dims and normalize if not done so already 
    a, α = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    mse_grad = Inf
    iter = 0
    grads = zeros(maxiters)
    losses = similar(grads)
    if use_E
        if E == nothing || Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=2)
                U , _ , V = svd(X .- Xavg)
                E, z, Y = Xavg , -V[:,1:K] , -U[:,1:K]
            else 
                E, z, Y = randn(rng,a,1), randn(rng,α, K), randn(rng,a, K)
            end
        end
        # create temps 
        E_tmp, z_tmp, Y_tmp = copy(E), copy(z), copy(Y)
        q = exp.(-Y*z' .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # iniital coef estimation 
        loss = tmi_loss(X,q) + 0.5*νz*sum(abs2, z) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            # update quantities
            dx = X-q
            z -= η*(dx'*Y_tmp + νz*z_tmp )
            Y -= η*(dx*z_tmp +  νy*Y_tmp)
            E -= η*sum(dx, dims=2) 
            
            # update q
            q = exp.(-Y*z' .- E) 
            q = normalize ? q ./ sum(q , dims = 1) : q
            # recompute error
            if print_err == true
                #err = 0.5*(sum(abs, λ - λ_tmp) + sum(abs, Y - Y_tmp))
                err_E = mean(abs, (E - E_tmp)/η)
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                loss = tmi_loss(X,q) + 0.5*νz*sum(abs2, z) + 0.5*νy*sum(abs2, Y)
                mse_grad = sqrt(err_E^2 + err_z^2 + err_Y^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = err
                losses[iter] = loss
            else
                err_E = mean(abs, (E - E_tmp)/η)
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                mse_grad = sqrt(err_E^2 + err_z^2 + err_Y^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
            end
            # update temps
            E_tmp , z_tmp, Y_tmp = E , z , Y
        end

        return  (E, z , Y, q) , (grads, losses)
    else 
        if Y == nothing || z == nothing
            if !use_rand_init
                U , _ , V = svd(X)
                z, Y = -V[:,1:K] , -U[:,1:K]
            else 
                z, Y = randn(α, K), randn(a, K)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        q = exp.(-Y*z') 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # initial coef estimation
        loss = tmi_loss(X,q) + 0.5*νz*sum(abs2, z) + 0.5*νy*sum(abs2, Y)
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
        end
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
            end
            # update quantities
            dx = X - q
            z -= η*(dx'*Y_tmp +  νz*z_tmp)
            Y -= η*(dx*z_tmp +  νy*Y_tmp)
            
            # update q
            q = exp.(-Y*z') 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2)
                loss = tmi_loss(X,q) + 0.5*νz*sum(abs2, z) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = mse_grad
                losses[iter] = loss
            else
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
            end
            # update temps 
            z_tmp, Y_tmp = z , Y
        end

        errs = (grads, losses)
        if !track_lats
            return (z , Y, q) ,  errs
        else 
            return (z , Y, q) , (zs, Ys) , errs
        end
    end
end

function tmi(X::Array{Float64, 3}, K::Int; 
    normalize=true,
    maxiters=100000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-5, # learning rate 
    use_rand_init=false,
    use_E=false, 
    E=nothing, 
    z=nothing, 
    Y=nothing, 
    track_lats = false,
    ρ = 0 , 
    ptol = 0.7,
    print_err = false,
    seed = 0
    )

    rng = Random.default_rng()
    Random.seed!(rng, seed)


    # get dims and normalize if not done so already 
    a, α, T = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    mse_grad = Inf
    iter = 0
    grads = zeros(maxiters)
    losses = similar(grads)
    if use_E
        if Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=(2,3))[:,1,1]
                z = zeros(α, K, T)
                Y = zeros(a,K,T)
                dX = X .- Xavg
                for t=1:T
                    U , _ , V = svd(dX[:,:,t]) 
                    z[:,:,t], Y[:,:,t] = -V[:,1:K] , -U[:,1:K]
                end
                E = Xavg
                Y = mean(Y, dims=3)[:,:,1]
            else
                z , Y = randn(rng,α,K,T), randn(rng,a,K)
                E = randn(rng,a)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        E_tmp = copy(E)
        @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
        q = exp.(-C .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # initial coef estimation
        
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
        end
        #loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
            end
            # update quantities
            dx = X - q

            # grads from reconstruction loss 
            @tensor dz[t,k,j] := dx[i,t,j]*Y_tmp[i,k]
            @tensor dY[a,k] := dx[a,t,j]*z_tmp[t,k,j]

            # update coeffs
            z -= η*dz
            Y -= η*dY
            E -= η*sum(dx, dims=(2,3))
            
            # update q
            @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
            q = exp.(-C .- E) 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_E= mean(abs, (E - E_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_E^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = err
                losses[iter] = loss
            else
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_E= mean(abs, (E - E_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 +  err_E^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
            end
            # update temps 
            E_tmp, z_tmp, Y_tmp = E, z , Y
        end
        errs = (grads, losses)
        if !track_lats
            return (E, z , Y, q) , errs
        else 
            return (E, z , Y, q) , Ξz, (zs, Ys) , errs
        end
    else 
        if Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=(2,3))[:,1,1]
                z = zeros(α, K, T)
                Y = zeros(a,K,T)
                dX = X .- Xavg
                for t=1:T
                    U , _ , V = svd(dX[:,:,t]) 
                    z[:,:,t], Y[:,:,t] = -V[:,1:K] , -U[:,1:K]
                end
                Y = mean(Y, dims=3)[:,:,1]
                
            else
                z , Y = rand(rng,α,K,T), rand(rng,a,K)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
        q = exp.(-C) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
        end
        #loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
            end
            # update quantities
            dx = X - q

            # grads from reconstruction loss 
            @tensor dz[t,k,j] := dx[i,t,j]*Y_tmp[i,k]
            @tensor dY[a,k] := dx[a,t,j]*z_tmp[t,k,j]
 
            # update latents 
            z -= η*dz
            Y -= η*dY
            
            # update q
            @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
            q = exp.(-C) 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = err
                losses[iter] = loss
            else
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
            end
            # update temps 
            z_tmp, Y_tmp = z , Y
        end
        errs = (grads, losses)
        if !track_lats
            return (z , Y, q) ,  errs
        else 
            return (z , Y, q) , (zs, Ys) , errs
        end
    end
end

function tmi(X::Array{Float64, 2}, K::Int, W, DW, N; 
    normalize=true,
    maxiters=100000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=false, 
    E=nothing, 
    z=nothing, 
    Y=nothing, 
    Ξz=nothing,
    Ξy=nothing,
    sparse_iter = 10000, 
    λz=1e1, # how strongly to enforce dzdt = θ(z)*Ξz
    νy = 0,
    νz = 0, 
    νz_sm = 0,
    cz = 1e-4, # adds small amount of noise to prevent overfitting, essentially ,
    track_lats = false,
    ρ = 0 , 
    ptol = 0.7,
    print_err = true,
    seed=0
    )

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    @assert 0 < sparse_iter < maxiters

    function Lib(z)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    # get dims and normalize if not done so already 
    a, α = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    mse_grad = Inf
    iter = 0
    grads = zeros(maxiters)
    losses = similar(grads)
    if use_E
        if E == nothing || Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=2)
                U , _ , V = svd(X .- Xavg)
                E, z, Y = Xavg , -V[:,1:K] , -U[:,1:K]
            else 
                E, z, Y = randn(rng,a,1), randn(rng,α, K), randn(rng,a, K)
            end
        end
        # create temps 
        E_tmp, z_tmp, Y_tmp = copy(E), copy(z), copy(Y)
        q = exp.(-Y*z' .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # iniital coef estimation 
        θ = Lib(z)
        Ξz = (W*θ)\(DW*z)
        Ξz_tmp = copy(Ξz)
        Ξz_active = abs.(Ξz) .> 0
        loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            # update quantities
            dzdt , θw =  DW*z_tmp , W*θ  
            dz = dzdt - θw*Ξz
            dx = X-q
            z -= η*(dx'*Y_tmp + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
            Y -= η*(dx*z_tmp + νy*Y_tmp)
            E -= η*sum(dx, dims=2) 
            #Ξz -= η*λz*(-θ'*W'*dz)
            if iter % sparse_iter == 0 && λz > 0
                # sparsify and get new active coefs 
                Ξz, _, _ = EnAdSR(θw, dzdt, "slic", tol=0.7, num_batches=10, c=cz, iter=10, Ξactive=Ξz_active) 
                Ξz_active = abs.(Ξz) .> 0
            else 
                # update active coefs 
                Ξz = update_active_coefs!(θw, dzdt, Ξz, Ξz_active)
            end
            # update q
            q = exp.(-Y*z' .- E) 
            q = normalize ? q ./ sum(q , dims = 1) : q
            if print_err == true
                # recompute error
                #err = 0.5*(sum(abs, λ - λ_tmp) + sum(abs, Y - Y_tmp))
                err_E = mean(abs, (E - E_tmp)/η)
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)

                mse_grad = sqrt(err_E^2 + err_z^2 + err_Y^2 + err_Ξz^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = mse_grad
                losses[iter] = loss
            else
                iter % 1000 == 0 ? println("iter => $(iter)") : nothing
            end
            # update temps
            E_tmp , z_tmp, Y_tmp, θ, Ξz_tmp = E , z , Y, Lib(z), Ξz
        end

        return  (E, z , Y, q) , Ξz, (grads, losses)
    else 
        if Y == nothing || z == nothing
            if !use_rand_init
                U , _ , V = svd(X)
                z, Y = -V[:,1:K] , -U[:,1:K]
            else 
                z, Y = randn(rng,α, K), randn(rng,a, K)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        q = exp.(-Y*z') 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # initial coef estimation
        θ = Lib(z)
        if Ξz == nothing
            Ξz = (W*θ)\(DW*z)
        end
        Ξactive = abs.(Ξz) .> 0
        Ξz_tmp = copy(Ξz)
        Ξz_active = abs.(Ξz) .> 0
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
            Ξzs = zeros(size(Ξz)..., maxiters)
        end
        loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
                Ξzs[:,:,iter] = Ξz
            end
            # update quantities
            dx = X - q
            dzdt , θw =  DW*z_tmp , W*θ 
            dz = dzdt - θw*Ξz
            z -= η*(dx'*Y_tmp + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
            Y -= η*(dx*z_tmp + νy*Y_tmp)
            
            if iter % sparse_iter == 0 && λz > 0
                # sparsify and get new active coefs 
                Ξz, _, _ = EnAdSR(θw, dzdt, "slic", tol=ptol, num_batches=10, c=cz, iter=5, ρ=ρ, Ξactive = Ξz_active) 
                Ξz_active = abs.(Ξz) .> 0
            else 
                # update active coefs 
                Ξz = update_active_coefs!(θw, dzdt, Ξz, Ξz_active)
            end
            
            # update q
            q = exp.(-Y*z') 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = err
                losses[iter] = loss
            end
            # update temps 
            z_tmp, Y_tmp, Ξz_tmp, θ = z , Y, Ξz, Lib(z)
        end
        errs = (grads, losses)
        if !track_lats
            return (z , Y, q) , Ξz, errs
        else 
            return (z , Y, q) , Ξz, (zs, Ys, Ξzs) , errs
        end
    end
end

function tmi(X::Array{Float64, 2}, K::Int, W, DW, N, θy; 
    normalize=true,
    maxiters=100000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    use_E=false, 
    E=nothing, 
    z=nothing, 
    Y=nothing, 
    Ξz=nothing,
    Ξy=nothing,
    sparse_iter = 10000, 
    λz=1e1, # how strongly to enforce dzdt = θz(z)*Ξz
    λy = 1e1, # how strongly to enforce Y = θy(features)*Ξy
    νy = 0,
    νz = 0, 
    νz_sm = 0,
    cz = 1e-4, # adds small amount of noise to prevent overfitting, essentially ,
    cy = 1e-4,
    track_lats = false,
    ρ = 0 , 
    ptol = 0.7, 
    print_err = true,
    use_cond=true,
    seed = 0
    )

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    @assert 0 < sparse_iter < maxiters

    function Lib(z)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    # get dims and normalize if not done so already 
    a, α = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    mse_grad = Inf
    iter = 0
    grads = zeros(maxiters)
    losses = similar(grads)
    if use_E
        if E == nothing || Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=2)
                U , _ , V = svd(X .- Xavg)
                E, z, Y = Xavg , -V[:,1:K] , -U[:,1:K]
            else 
                E, z, Y = randn(rng,a,1), randn(rng,α, K), randn(rng,a, K)
            end
        end
        # create temps 
        E_tmp, z_tmp, Y_tmp = copy(E), copy(z), copy(Y)
        q = exp.(-Y*z' .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # iniital coef estimation 
        θ = Lib(z)
        Ξz = (W*θ)\(DW*z)
        Ξz_tmp = copy(Ξz)
        Ξy = θy\Y
        Ξy_tmp = copy(Ξy)
        Ξz_active = abs.(Ξz) .> 0
        Ξy_active = abs.(Ξy) .> 0
        loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,Y, θy, Ξy, λz,νz,νz_sm, λy,νy)
        while iter < maxiters && mse_grad > tol
            iter += 1
            # update quantities
            dzdt , θw =  DW*z_tmp , W*θ 
            dz = dzdt - θw*Ξz
            dx = X-q
            z -= η*(dx'*Y_tmp + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
            Y -= η*(dx*z_tmp + λy*(-θy*Ξy + Y_tmp) + νy*Y_tmp)
            E -= η*sum(dx, dims=2) 
            #Ξz -= η*λz*(-θ'*W'*dz)
            if iter % sparse_iter == 0 && λz > 0
                # sparsify and get new active coefs 
                Ξz, _, _ = EnAdSR(θw, dzdt, "slic", tol=0.7, num_batches=10, c=cz, iter=10, Ξactive=Ξz_active) 
                Ξz_active = abs.(Ξz) .> 0
                Ξy, _, _ = EnAdSR(θy, Y_tmp, "slic", tol=0.7, num_batches=10, c=cy, iter=10, Ξactive=Ξy_active) 
                Ξy_active = abs.(Ξy) .> 0
            else 
                # update active coefs 
                Ξz = update_active_coefs!(θw, dzdt, Ξz, Ξz_active)
                Ξy = update_active_coefs!(θy, Y_tmp, Ξy, Ξy_active)
            end
            # update q
            q = exp.(-Y*z' .- E) 
            q = normalize ? q ./ sum(q , dims = 1) : q
            if print_err == true
                # recompute error
                #err = 0.5*(sum(abs, λ - λ_tmp) + sum(abs, Y - Y_tmp))
                err_E = mean(abs, (E - E_tmp)/η)
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)
                err_Ξy = mean(abs, (Ξy - Ξy_tmp)/η)

                mse_grad = sqrt(err_E^2 + err_z^2 + err_Y^2 + err_Ξz^2 + err_Ξy^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,Y, θy, Ξy, λz,νz,νz_sm, λy,νy)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = mse_grad
                losses[iter] = loss
            else
                iter % 1000 == 0 ? println("iter => $(iter)") : nothing
            end
            # update temps
            E_tmp , z_tmp, Y_tmp, θ, Ξz_tmp, Ξy_tmp = E , z , Y, Lib(z), Ξz, Ξy
        end

        return  (E, z , Y, q) , (Ξz, Ξy), (grads, losses)
    else 
        if Y == nothing || z == nothing
            if !use_rand_init
                U , _ , V = svd(X)
                z, Y = -V[:,1:K] , -U[:,1:K]
            else 
                #z, Y = randn(rng,α, K), randn(rng,a, K)
                z, Y = rand(rng,α, K), rand(rng,a, K)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        q = exp.(-Y*z') 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # initial coef estimation
        θ = Lib(z)
        if Ξz == nothing
            Ξz = (W*θ)\(DW*z)
        end
        Ξactive = abs.(Ξz) .> 0
        Ξz_tmp = copy(Ξz)
        if Ξy == nothing 
            Ξy = θy\Y
        end
        #print(Ξz)
        #print(Ξy)
        Ξy_tmp = copy(Ξy)
        Ξz_active = abs.(Ξz) .> 0
        Ξy_active = abs.(Ξy) .> 0
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
            Ξzs = zeros(size(Ξz)..., maxiters)
            Ξys = zeros(size(Ξy)..., maxiters)
        end
        loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,Y, θy, Ξy, λz,νz,νz_sm, λy,νy)
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
                Ξzs[:,:,iter] = Ξz
                Ξys[:,:,iter] = Ξy
            end
            # update quantities
            dx = X - q
            dzdt , θw =  DW*z_tmp , W*θ 
            dz = dzdt - θw*Ξz
            z -= η*(dx'*Y_tmp + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
            Y -= η*(dx*z_tmp + λy*(-θy*Ξy + Y_tmp)+ νy*Y_tmp)
            
            if iter % sparse_iter == 0 && λz > 0
                # sparsify and get new active coefs 
                Ξz, _, _ = EnAdSR(θw, dzdt, "slic", tol=ptol, num_batches=10, c=cz, iter=5, ρ=ρ, Ξactive = Ξz_active, use_cond=use_cond) 
                Ξz_active = abs.(Ξz) .> 0
                Ξy, _, _ = EnAdSR(θy, Y_tmp, "slic", tol=ptol, num_batches=10, c=cy, iter=5, ρ=ρ, Ξactive = Ξy_active, use_cond=use_cond) 
                Ξy_active = abs.(Ξy) .> 0
            else 
                # update active coefs 
                Ξz = update_active_coefs!(θw, dzdt, Ξz, Ξz_active)
                Ξy = update_active_coefs!(θy, Y_tmp, Ξy, Ξy_active)
            end
            
            # update q
            q = exp.(-Y*z') 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)
                err_Ξy = mean(abs, (Ξy - Ξy_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2 + err_Ξz^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,Y, θy, Ξy, λz,νz,νz_sm, λy,νy)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = mse_grad
                losses[iter] = loss
            else
                iter % 1000 == 0 ? println("iter => $(iter)") : nothing
            end
            # update temps 
            z_tmp, Y_tmp, Ξz_tmp, Ξy_tmp, θ = z , Y, Ξz, Ξy, Lib(z)
        end
        errs = (grads, losses)
        if !track_lats
            return (z , Y, q) , (Ξz, Ξy), errs
        else 
            return (z , Y, q) , (Ξz, Ξy), (zs, Ys, Ξzs, Ξys) , errs
        end
    end
end

# if we have trials where we want to restrict the 
function tmi(X::Array{Float64, 3}, K::Int, W, DW, N; 
    normalize=true,
    maxiters=100000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-5, # learning rate 
    use_rand_init=false,
    use_E=false, 
    E=nothing, 
    z=nothing, 
    Y=nothing, 
    Ξz=nothing,
    sparse_iter = 10000, 
    λz=1e1, # how strongly to enforce dzdt = θ(z)*Ξz
    νy = 0,
    νz = 0, 
    νz_sm = 0,
    cz = 1e-6, # adds small amount of noise to prevent overfitting, essentially ,
    track_lats = false,
    ρ = 0 , 
    ptol = 0.7,
    print_err = false,
    seed = 0
    )

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    #@assert 0 < sparse_iter < maxiters

    
    function Lib(z::Array{Float64,2})
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end
     
    function Lib(z::Array{Float64,3})
        @assert size(z,2) == size(N,2)
        T, L , trials = size(z,1), size(N,1), size(z,3)
        θ = zeros(T,L,trials)
        for trial=1:trials
            for l=1:L
                θ[:,l,trial] = ones(T)
                for s=1:size(N,2)
                    θ[:,l,trial] = θ[:,l,trial].*z[:,s,trial].^N[l,s]
                end
            end
        end
        return θ
    end

    # get dims and normalize if not done so already 
    a, α, T = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    mse_grad = Inf
    iter = 0
    grads = zeros(maxiters)
    losses = similar(grads)
    if use_E
        if Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=(2,3))[:,1,1]
                z = zeros(α, K, T)
                Y = zeros(a,K,T)
                dX = X .- Xavg
                for t=1:T
                    U , _ , V = svd(dX[:,:,t]) 
                    z[:,:,t], Y[:,:,t] = -V[:,1:K] , -U[:,1:K]
                end
                E = Xavg
                Y = mean(Y, dims=3)[:,:,1]
            else
                z , Y = randn(rng,α,K,T), randn(rng,a,K)
                E = randn(rng,a)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        E_tmp = copy(E)
        @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
        q = exp.(-C .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # initial coef estimation
        θ = Lib(z)
        print(size(θ))
        if Ξz == nothing
            @tensor dzdt[t,k,j] := DW[t,l]*z_tmp[l,k,j]
            @tensor θw[t,k,j] := W[t,l]*θ[l,k,j]
            # reshape tensors and estimate model 
            dzdt_rs, θw_rs = ReshapeMats(θw, dzdt)
            #Ξz = reshape(θw, (size(θw,1)*size(θw,3), size(θw,2)))\reshape(dzdt, (size(dzdt,1)*size(dzdt,3), size(dzdt,2)))
            Ξz = θw_rs \ dzdt_rs
        end
        Ξz_tmp = copy(Ξz)
        Ξz_active = abs.(Ξz) .> 0
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
            Ξzs = zeros(size(Ξz)..., maxiters)
        end
        #loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
                Ξzs[:,:,iter] = Ξz
            end
            # update quantities
            dx = X - q

            # grads from reconstruction loss 
            @tensor dz[t,k,j] := dx[i,t,j]*Y_tmp[i,k]
            @tensor dY[a,k] := dx[a,t,j]*z_tmp[t,k,j]

            # CREATE LIBRARY FROM ZS???
            # (FILL)
            
            # grads from model loss 
            @tensor dzdt[t,k,j] := DW[t,l]*z_tmp[l,k,j]
            @tensor θw[t,k,j] := W[t,l]*θ[l,k,j]
            @tensor Δz[t,k,j] := dzdt[t,k,j] - θw[t,l,j]*Ξz[l,k]

            # computing terms in model loss 
            @tensor dz_12[t,k,j] := DW[l,t]*Δz[l,k,j] 
            @tensor dz_22[t,k,j] := -W[l,t]*Δz[l,n,j]*Ξz[k,n]
            dz_22 = dz_22 .* θ
            @tensor dz_23[t,k,j] := dz_22[t,l,j]*N[l,k]
            dz_23 = dz_23 ./ (z_tmp .+ 1e-8)
            dz = dz + λz*(dz_12 + dz_23)

            #=
            z -= η*(dx'*Y_tmp + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
            Y -= η*(dx*z_tmp + νy*Y_tmp)
            =# 
            # update latents 
            z -= η*dz
            Y -= η*dY
            E -= η*sum(dx, dims=(2,3))
            # compute model 
            #=
            θw_rs = reshape(θw, (size(θw,1)*size(θw,3), size(θw,2)))
            dzdt_rs = reshape(dzdt, (size(dzdt,1)*size(dzdt,3), size(dzdt,2)))
            =# 
            dzdt_rs, θw_rs = ReshapeMats(θw, dzdt)
            if iter % sparse_iter == 0 && λz > 0
                # sparsify and get new active coefs 
                Ξz, _, _ = EnAdSR(θw_rs, dzdt_rs, "slic", tol=ptol, num_batches=10, c=cz, iter=5, ρ=ρ, Ξactive = Ξz_active) 
                Ξz_active = abs.(Ξz) .> 0
            else 
                # update active coefs 
                Ξz = update_active_coefs!(θw_rs, dzdt_rs, Ξz, Ξz_active)
            end
            
            # update q
            @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
            q = exp.(-C .- E) 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = err
                losses[iter] = loss
            else
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)
                err_E= mean(abs, (E - E_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2 + err_E^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
            end
            # update temps 
            E_tmp, z_tmp, Y_tmp, Ξz_tmp, θ = E, z , Y, Ξz, Lib(z)
        end
        errs = (grads, losses)
        if !track_lats
            return (E, z , Y, q) , Ξz, errs
        else 
            return (E, z , Y, q) , Ξz, (zs, Ys, Ξzs) , errs
        end
    else 
        if Y == nothing || z == nothing
            if !use_rand_init
                Xavg = mean(X, dims=(2,3))[:,1,1]
                z = zeros(α, K, T)
                Y = zeros(a,K,T)
                dX = X .- Xavg
                for t=1:T
                    U , _ , V = svd(dX[:,:,t]) 
                    z[:,:,t], Y[:,:,t] = -V[:,1:K] , -U[:,1:K]
                end
                Y = mean(Y, dims=3)[:,:,1]
                
            else
                z , Y = rand(rng,α,K,T), rand(rng,a,K)
            end
        end
        
        z_tmp, Y_tmp = copy(z), copy(Y)
        @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
        q = exp.(-C) 
        q = normalize ? q ./ sum(q , dims = 1) : q
        # initial coef estimation
        θ = Lib(z)
        if Ξz == nothing
            @tensor dzdt[t,k,j] := DW[t,l]*z_tmp[l,k,j]
            @tensor θw[t,k,j] := W[t,l]*θ[l,k,j]
            # reshape tensors and estimate model 
            Ξz = reshape(θw, (size(θw,1)*size(θw,3), size(θw,2)))\reshape(dzdt, (size(dzdt,1)*size(dzdt,3), size(dzdt,2)))
        end
        Ξz_tmp = copy(Ξz)
        Ξz_active = abs.(Ξz) .> 0
        if track_lats
            zs = zeros(size(z)..., maxiters)
            Ys = zeros(size(Y)..., maxiters)
            Ξzs = zeros(size(Ξz)..., maxiters)
        end
        #loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
        while iter < maxiters && mse_grad > tol
            iter += 1
            if track_lats
                zs[:,:,iter] = z 
                Ys[:,:,iter] = Y 
                Ξzs[:,:,iter] = Ξz
            end
            # update quantities
            dx = X - q

            # grads from reconstruction loss 
            @tensor dz[t,k,j] := dx[i,t,j]*Y_tmp[i,k]
            @tensor dY[a,k] := dx[a,t,j]*z_tmp[t,k,j]

            # CREATE LIBRARY FROM ZS???
            # (FILL)

            # grads from model loss 
            @tensor dzdt[t,k,j] := DW[t,l]*z_tmp[l,k,j]
            @tensor θw[t,k,j] := W[t,l]*θ[l,k,j]
            @tensor Δz[t,k,j] := dzdt[t,k,j] - θw[t,l,j]*Ξz[l,k]

            # computing terms in model loss 
            @tensor dz_12[t,k,j] := DW[l,t]*Δz[l,k,j] 
            @tensor dz_22[t,k,j] := -W[l,t]*Δz[l,n,j]*Ξz[k,n]
            dz_22 = dz_22 .* θ
            @tensor dz_23[t,k,j] := dz_22[t,l,j]*N[l,k]
            dz_23 = dz_23 ./ (z_tmp .+ 1e-8)
            dz = dz + λz*(dz_12 + dz_23)

            #=
            z -= η*(dx'*Y_tmp + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
            Y -= η*(dx*z_tmp + νy*Y_tmp)
            =# 
            # update latents 
            z -= η*dz
            Y -= η*dY

            # compute model 
            θw_rs = reshape(θw, (size(θw,1)*size(θw,3), size(θw,2)))
            dzdt_rs = reshape(dzdt, (size(dzdt,1)*size(dzdt,3), size(dzdt,2)))
            if iter % sparse_iter == 0 && λz > 0
                # sparsify and get new active coefs 
                Ξz, _, _ = EnAdSR(θw_rs, dzdt_rs, "slic", tol=ptol, num_batches=10, c=cz, iter=5, ρ=ρ, Ξactive = Ξz_active) 
                Ξz_active = abs.(Ξz) .> 0
            else 
                # update active coefs 
                Ξz = update_active_coefs!(θw_rs, dzdt_rs, Ξz, Ξz_active)
            end
            
            # update q
            @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
            q = exp.(-C) 
            q = normalize ? q ./ sum(q , dims = 1) : q

            if print_err == true
                # compute mean derivative estimates
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2)
                loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad), loss => $(loss)") : nothing
                grads[iter] = err
                losses[iter] = loss
            else
                err_z = mean(abs, (z - z_tmp)/η) 
                err_Y = mean(abs, (Y - Y_tmp)/η)
                err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)

                mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2)
                iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
            end
            # update temps 
            z_tmp, Y_tmp, Ξz_tmp, θ = z , Y, Ξz, Lib(z)
        end
        errs = (grads, losses)
        if !track_lats
            return (z , Y, q) , Ξz, errs
        else 
            return (z , Y, q) , Ξz, (zs, Ys, Ξzs) , errs
        end
    end
end




function ReshapeMats(θ::Array{Float64, 3}, dzdt::Array{Float64, 3})
    @assert size(θ,3) == size(dzdt,3)
    θ_rs = θ[:,:,1]
    dzdt_rs = dzdt[:,:,1]

    for i=1:size(θ,3)
        θ_rs = vcat(θ_rs, θ[:,:,i])
        dzdt_rs =  vcat(dzdt_rs, dzdt[:,:,i])
    end

    return dzdt_rs, θ_rs
end

# updates the nonzero model terms 
function update_active_coefs!(θ, dz, Ξz, Ξactive)
    # final regression to update
    n_state = size(dz, 2)
    Ξz = Ξz .* Ξactive
    for ind=1:n_state
        biginds = Ξactive[:,ind]
        Ξz[biginds,ind] = θ[:,biginds]\dz[:,ind]
    end 

    return Ξz
end

# Losses
# returns KLD
function tmi_loss(p::Array{Float64, 2}, q::Array{Float64, 2}; eps = 1e-8)
    pn = p ./ sum(p, dims=1)
    qn = q ./ sum(q, dims=1)
    return sum(pn .* log.((pn .+ eps) ./ (qn .+ eps)))    
end

function tmi_loss(p::Array{Float64, 3}, q::Array{Float64, 3}; eps = 1e-8)
    n = size(p,3)
    kld = 0 
    for i=1:n 
        kld += tmi_loss(p[:,:,i], q[:,:,i]; eps = eps)
    end
    
    return kld/n   
end

# returns KLD + z model loss + reg terms
function tmi_loss(p::Array{Float64, 2}, q::Array{Float64, 2}, DW, W , z, Lib::Function, Ξz, λz, νz, νz_sm)
    kld_loss = tmi_loss(p,q)
    model_loss = 0.5*λz*sum(abs2, DW*z - W*Lib(z)*Ξz)
    L2_z_reg_loss = 0.5*νz*sum(abs2, z)
    L2_smooth_reg_loss = 0.5*νz_sm*sum(abs2, DW*z)
    return kld_loss + model_loss + L2_z_reg_loss + L2_smooth_reg_loss
end
# returns KLD + z model loss + Y model loss + reg terms
function tmi_loss(p::Array{Float64, 2}, q::Array{Float64, 2}, DW, W , z, Lib, Ξz, Y, θy, Ξy, λz, νz, νz_sm, λy, νy)
    loss_pqz = tmi_loss(p, q, DW, W , z, Lib, Ξz, λz, νz, νz_sm)
    Y_model_loss =  0.5*λy*sum(abs2, Y - θy*Ξy)
    L2_Y_reg_loss = 0.5*νy*sum(abs2, Y)
    return loss_pqz + Y_model_loss + L2_Y_reg_loss
end


# ================================= Hyperparam opt ========================================

function tmi_w_hp_scan(X::Array{Float64, 2}, K::Int, W, DW, N, θy, λzs::AbstractVector, λys::AbstractVector; 
    normalize=true,
    maxiters=100000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=false,
    sparse_iter = 10000, 
    cz = 1e-6, # adds small amount of noise to prevent overfitting, essentially ,
    cy = 1e1, 
    use_cond =true,
    seed=0,
    z=nothing,
    Y=nothing
    )

    models = []

    for λz in λzs 
        for λy in λys 
            println("λz => $(λz), λy => $(λy)")
            (z, Y, q), (Ξz, Ξy), _ = tmi(X, K, W, DW, N, θy; 
                normalize=normalize,
                maxiters=maxiters, # max number of iterations to update params 
                tol=tol, # termination condition if total L2 error of derivatives across iter < tol
                η=η, # learning rate 
                use_rand_init=use_rand_init,
                sparse_iter = sparse_iter, 
                λz = λz, # how strongly to enforce dzdt = θz(z)*Ξz
                λy = λy, # how strongly to enforce Y = θy(features)*Ξy
                cz = cz, # adds small amount of noise to prevent overfitting, essentially ,
                cy = cy,
                print_err = false,
                use_cond =use_cond,
                seed=seed,
                z=z,
                Y=Y 
            )
            model = Dict("z" => z, "Y" => Y , "q" => q, "Ξz" => Ξz, "Ξy" => Ξy, "λz" => λz, "λy" => λy)
            push!(models, model)
        end
    end

    return models 
end

function tmi_w_hp_scan(X::Array{Float64, 3}, K::Int, W, DW, N, λzs::AbstractVector; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-3, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-3, # learning rate 
    use_rand_init=true,
    sparse_iter = 50000, 
    cz = 1e-6, # adds small amount of noise to prevent overfitting, essentially ,
    use_cond =true,
    use_E = true
    )

    models = []

    if use_E
        for λz in λzs 
            println("λz => $(λz)")
            (E, z, Y, q), Ξz, _ = tmi(X, K, W, DW, N; 
                maxiters=maxiters, # max number of iterations to update params 
                tol=tol, # termination condition if total L2 error of derivatives across iter < tol
                η=η, # learning rate 
                use_rand_init=use_rand_init,
                sparse_iter = sparse_iter, 
                λz = λz, # how strongly to enforce dzdt = θz(z)*Ξz
                cz = cz, # adds small amount of noise to prevent overfitting, essentially ,
                use_E=use_E
            )
            model = Dict("E" => E, "z" => z, "Y" => Y , "q" => q, "Ξz" => Ξz, "λz" => λz)
            push!(models, model)

        end
    else
        for λz in λzs 
            println("λz => $(λz)")
            (z, Y, q), Ξz, _ = tmi(X, K, W, DW, N; 
                maxiters=maxiters, # max number of iterations to update params 
                tol=tol, # termination condition if total L2 error of derivatives across iter < tol
                η=η, # learning rate 
                use_rand_init=use_rand_init,
                sparse_iter = sparse_iter, 
                λz = λz, # how strongly to enforce dzdt = θz(z)*Ξz
                cz = cz, # adds small amount of noise to prevent overfitting, essentially ,
                use_E=use_E
            )
            model = Dict("z" => z, "Y" => Y , "q" => q, "Ξz" => Ξz, "λz" => λz)
            push!(models, model)

        end
    end

    return models 
end

function score_models(models, N, θy, W, DW)

    function Lib(z, N)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    function slic_score(Ξz, z, W, DW) 
        n = size(z,1)
        k = count(abs.(Ξz) .> 0) + 1
        RSS = sum(abs2, DW*z - W*Lib(z, N)*Ξz)
        return n*log(k*RSS/n)
    end 

    function slic_score(Ξy, Y, θy) 
        n = size(Y,1)
        k = count(abs.(Ξy) .> 0) + 1
        RSS = sum(abs2, Y - θy*Ξy)
        return n*log(k*RSS/n)
    end 

    #scores_pred = []
    scores_ic = []
    for model in models 
        Ξz = model["Ξz"]
        Ξy = model["Ξy"]
        z = model["z"]
        Y = model["Y"]
        score_z = slic_score(Ξz, z, W, DW)
        score_Y = slic_score(Ξy, Y, θy)
        score_total = score_z + score_Y
        push!(scores_ic, score_total)
        #=
        Ypred = θy*Ξy
        zpred = PredTraj(N, Ξy, )
        =# 
    end

    return scores_ic
end

function score_models(models, N, W, DW)

    function Lib(z::Array{Float64, 2}, N)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    function Lib(z::Array{Float64,3}, N)
        @assert size(z,2) == size(N,2)
        T, L , trials = size(z,1), size(N,1), size(z,3)
        θ = zeros(T,L,trials)
        for trial=1:trials
            for l=1:L
                θ[:,l,trial] = ones(T)
                for s=1:size(N,2)
                    θ[:,l,trial] = θ[:,l,trial].*z[:,s,trial].^N[l,s]
                end
            end
        end
        return θ
    end

    function slic_score(Ξz, dz, θ) 
        n = size(dz,1)
        k = count(abs.(Ξz) .> 0) + 1
        RSS = sum(abs2, dz - θ*Ξz)
        return n*log(k*RSS/n)
    end 

    #scores_pred = []
    scores_ic = []
    for model in models 
        Ξz = model["Ξz"]
        z = model["z"]
        θ = Lib(z, N)
        @tensor dz[t,k,j] := DW[t,l]*z[l,k,j]
        @tensor θw[t,k,j] := W[t,l]*θ[l,k,j]
        dz, θ = ReshapeMats(θw, dz)
        score_z = slic_score(Ξz, dz,θ)
        push!(scores_ic, score_z)
    end

    return scores_ic
end

function latent_dim_scan(X, Ks::Vector; normalize=true, tol=1e-5, η=1e-3, maxiters=100000, use_E=false)
    qs = []
    klds = []
    for K in Ks
        println("K => $(K)")
        if use_E
            (_,_,_, q), _ = tmi(X, K; maxiters=maxiters, η=η, tol=tol, use_E=true, print_err=false, normalize=normalize)
            kld = tmi_loss(X, q)
            push!(qs,q)
            push!(klds, kld)
        else
            (_,_, q), _ = tmi(X, K; maxiters=maxiters, η=η, tol=tol, use_E=false, print_err=false, normalize=normalize)
            kld = tmi_loss(X, q)
            push!(qs,q)
            push!(klds, kld)
        end
    end
    return qs, klds
end


# ================================= Get Z0 and update Ξz ========================================

function get_z0_and_update_model(X::Array{Float64, 2}, Y, Ξz, N, DW, W; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-6, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-1, # learning rate 
    use_rand_init=false,
    νz=0., 
    νz_sm = 0.,
    λz = 1e1
    )

    function Lib(z)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    # get dims and normalize if not done so already 
    a, α = size(X)
    K = size(Y, 2)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    err = Inf
    iter = 0
    
    # Initialize z
    z = begin
        if !use_rand_init
            U , _ , V = svd(X)
            -V[:,1:K] 
        else 
            randn(α, K)
        end
    end
    
    z_tmp = copy(z)
    q = exp.(-Y*z') 
    q = normalize ? q ./ sum(q , dims = 1) : q
    # initial coef estimation
    θ = Lib(z_tmp)
    Ξz_active = abs.(Ξz) .> 0
    while iter < maxiters && err > tol
        iter += 1
        # update z
        dx = X - q
        #z -= η*dx'*Y 
        dzdt , θw =  DW*z_tmp , W*θ 
        dz = dzdt - θw*Ξz
        z -= η*(dx'*Y + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
        Ξz = update_active_coefs!(θw, dzdt, Ξz, Ξz_active)
        # update q
        q = exp.(-Y*z') 
        q = normalize ? q ./ sum(q , dims = 1) : q

        # compute mean change in z 
        err = mean(abs, (z - z_tmp)/η)
        # break if beneath tol 
        #err < tol ? break : continue
        # print 
        iter % 1000 == 0 ? println("iter => $(iter) , err => $(err)") : nothing
        # update temps
        z_tmp, θ = z, Lib(z)
    end
    return z, Ξz, q
end

function get_z0_and_update_model(X::Array{Float64, 2}, Y, E, Ξz, N, DW, W; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-6, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-1, # learning rate 
    use_rand_init=false,
    νz=0., 
    νz_sm = 0.,
    λz = 1e1,
    z = nothing
    )

    function Lib(z)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    # get dims and normalize if not done so already 
    a, α = size(X)
    K = size(Y, 2)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    err = Inf
    iter = 0
    
    # Initialize z
    if isequal(z,nothing)
        z = begin
            if !use_rand_init
                U , _ , V = svd(X)
                -V[:,1:K] 
            else 
                randn(α, K)
            end
        end
    end
    
    z_tmp = copy(z)
    q = exp.(-Y*z' .- E) 
    q = normalize ? q ./ sum(q , dims = 1) : q
    # initial coef estimation
    θ = Lib(z_tmp)
    Ξz_active = abs.(Ξz) .> 0
    while iter < maxiters && err > tol
        iter += 1
        # update z
        dx = X - q
        #z -= η*dx'*Y 
        dzdt , θw =  DW*z_tmp , W*θ 
        dz = dzdt - θw*Ξz
        z -= η*(dx'*Y + λz*(DW'*dz - ((W'*dz*Ξz') .* θ)*N./(z_tmp .+ 1e-8)) + νz*z_tmp + νz_sm*DW'*DW*z_tmp)
        Ξz = update_active_coefs!(θw, dzdt, Ξz, Ξz_active)
        # update q
        q = exp.(-Y*z' .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q

        # compute mean change in z 
        err = mean(abs, (z - z_tmp)/η)
        # break if beneath tol 
        #err < tol ? break : continue
        # print 
        iter % 1000 == 0 ? println("iter => $(iter) , err => $(err)") : nothing
        # update temps
        z_tmp, θ = z, Lib(z)
    end
    return z, Ξz, q
end

function get_z0_and_update_model(X::Array{Float64, 3}, Y, E, Ξz, N, DW, W; 
    normalize=true,
    maxiters=150000, # max number of iterations to update params 
    tol=1e-6, # termination condition if total L2 error of derivatives across iter < tol
    η=1e-1, # learning rate 
    use_rand_init=false,
    νz=0., 
    νz_sm = 0.,
    λz = 1e1,
    z = nothing
    )

    function Lib(z)
        @assert size(z,2) == size(N,2)
        T, L = size(z,1), size(N,1)
        θ = zeros(T,L)
        for l=1:L
            θ[:,l] = ones(T)
            for s=1:size(N,2)
                θ[:,l] = θ[:,l].*z[:,s].^N[l,s]
            end
        end

        return θ
    end

    function Lib(z::Array{Float64,3})
        @assert size(z,2) == size(N,2)
        T, L , trials = size(z,1), size(N,1), size(z,3)
        θ = zeros(T,L,trials)
        for trial=1:trials
            for l=1:L
                θ[:,l,trial] = ones(T)
                for s=1:size(N,2)
                    θ[:,l,trial] = θ[:,l,trial].*z[:,s,trial].^N[l,s]
                end
            end
        end
        return θ
    end

    # get dims and normalize if not done so already 
    a, α, T = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    mse_grad = Inf
    iter = 0
    
    if isequal(z,nothing)
        if !use_rand_init
            Xavg = mean(X, dims=(2,3))[:,1,1]
            z = zeros(α, K, T)
            Y = zeros(a,K,T)
            dX = X .- Xavg
            for t=1:T
                U , _ , V = svd(dX[:,:,t]) 
                z[:,:,t] = -V[:,1:K] 
            end
           
        else
            z = randn(rng,α,K,T)
        end
    end
    
    z_tmp = copy(z)
    @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
    q = exp.(-C .- E) 
    q = normalize ? q ./ sum(q , dims = 1) : q
    # initial coef estimation
    θ = Lib(z)
    
    Ξz_tmp = copy(Ξz)
    Ξz_active = abs.(Ξz) .> 0
    
    #loss = tmi_loss(X,q,DW,W,z,Lib,Ξz,λz,νz,νz_sm) + 0.5*νy*sum(abs2, Y)
    while iter < maxiters && mse_grad > tol
        iter += 1
        
        # update quantities
        dx = X - q

        # grads from reconstruction loss 
        @tensor dz[t,k,j] := dx[i,t,j]*Y_tmp[i,k]
        
        # grads from model loss 
        @tensor dzdt[t,k,j] := DW[t,l]*z_tmp[l,k,j]
        @tensor θw[t,k,j] := W[t,l]*θ[l,k,j]
        @tensor Δz[t,k,j] := dzdt[t,k,j] - θw[t,l,j]*Ξz[l,k]

        # computing terms in model loss 
        @tensor dz_12[t,k,j] := DW[l,t]*Δz[l,k,j] 
        @tensor dz_22[t,k,j] := -W[l,t]*Δz[l,n,j]*Ξz[k,n]
        dz_22 = dz_22 .* θ
        @tensor dz_23[t,k,j] := dz_22[t,l,j]*N[l,k]
        dz_23 = dz_23 ./ (z_tmp .+ 1e-8)
        dz = dz + λz*(dz_12 + dz_23)

    
        # update z 
        z -= η*dz
        # update model 
        dzdt_rs, θw_rs = ReshapeMats(θw, dzdt)
        Ξz = update_active_coefs!(θw_rs, dzdt_rs, Ξz, Ξz_active)
        
        # update q
        @tensor C[a,t,j] := Y[a, k]*z[t,k,j]
        q = exp.(-C .- E) 
        q = normalize ? q ./ sum(q , dims = 1) : q

        
        err_z = mean(abs, (z - z_tmp)/η) 
        err_Y = mean(abs, (Y - Y_tmp)/η)
        err_Ξz = mean(abs, (Ξz - Ξz_tmp)/η)
        err_E= mean(abs, (E - E_tmp)/η)

        mse_grad = sqrt(err_z^2 + err_Y^2 + err_Ξz^2 + err_E^2)
        iter % 1000 == 0 ? println("iter => $(iter) , mse_grad => $(mse_grad)") : nothing
        
        # update temps 
        z_tmp, Ξz_tmp, θ = z , Ξz, Lib(z)
    end
    
    return (z , q) , Ξz
    
end