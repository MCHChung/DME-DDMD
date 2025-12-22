using LinearAlgebra, StatsBase

function nnmf(X::Array{Float64,2}, K::Int; 
    maxiters=100000, 
    use_rand_init=false, 
    normalize=true,
    loss_thresh=Inf,
    grad_thresh = 1e-6,
    eps=1e-8
    )

    a,α = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    if !use_rand_init
        U , _ , V = svd(X)
        B, A = abs.(V[:,1:K]) , abs.(U[:,1:K])
    else 
        B, A = rand(α, K), rand(a, K)
    end
    #A = A ./ sum(A, dims=1)
    A_tmp, B_tmp = copy(A), copy(B)
    q = A*B'
    q = q ./ sum(q, dims=1)
    loss = kld_loss(X, q)
    grad_norm = Inf
    iter = 0
    while iter < maxiters && loss < loss_thresh && grad_norm > grad_thresh
        # multiplicative update scheme from https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf 
        qu = 1 ./ (q .+ eps)
        A = (A_tmp .* ((X .* qu) * B_tmp)) ./ sum(B_tmp, dims=1)
        B = (B_tmp .* ((X .* qu)' * A_tmp)) ./ sum(A_tmp, dims=1)

        # grad norm 
        mean_grad_A = mean(abs, A - A_tmp)
        mean_grad_B = mean(abs, B - B_tmp)
        grad_norm = sqrt(mean_grad_A^2 + mean_grad_B^2)
        
        # update 
        q = A*B'
        q = q ./ sum(q, dims=1)
        A_tmp, B_tmp = A, B

        # compute loss 
        loss = kld_loss(X,q)
        iter % 1000 == 0 ? println("Iter => $(iter), grad_norm => $(grad_norm), kld_loss => $(loss) ") : nothing
        iter += 1
    end

    return A, B, q
end

function nnmf(X::Array{Float64,3}, K::Int; 
    maxiters=100000, 
    use_rand_init=false, 
    normalize=true,
    loss_thresh=Inf,
    grad_thresh = 1e-6,
    eps=1e-8
    )

    a,α,T = size(X)
    if normalize
        X = X ./ sum(X, dims=1)
    end

    As = zeros(a,K,T)
    Bs = zeros(α, K, T)
    qs = similar(X)
    for i=1:T
        A,B,q = nnmf(X[:,:,i], K; 
        maxiters=maxiters, 
        use_rand_init=use_rand_init, 
        normalize=normalize,
        loss_thresh=loss_thresh,
        grad_thresh = grad_thresh,
        eps=eps
        )
        As[:,:,i] = A
        Bs[:,:,i] = B
        qs[:,:,i] = q
    end
    return As, Bs, qs
end

function kld_loss(p::Array{Float64, 2}, q::Array{Float64, 2}; eps = 1e-8)
    pn = p ./ sum(p, dims=1)
    qn = q ./ sum(q, dims=1)
    return sum(pn .* log.((pn .+ eps) ./ (qn .+ eps)))    
end

# if there are trials (along 3rd axis), return the sum of all klds across trials 
function kld_loss(p::Array{Float64, 3}, q::Array{Float64, 3}; eps = 1e-8)
    kld = 0 
    for i=1:size(p,3)
        kld_i = kld_loss(p[:,:,i], q[:,:,i], eps=eps)
        kld += kld_i
    end
    
    return kld
end