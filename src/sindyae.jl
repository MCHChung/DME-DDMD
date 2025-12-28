using LuxCore, WeightInitializers, Random

struct SINDyAE{L1, L2, L3} <: LuxCore.AbstractLuxContainerLayer{(:enc, :dec, :sindy)}
    enc::L1
    dec::L2
    sindy::L3
  end 


struct SINDyLayer{F1, F2} <: LuxCore.AbstractLuxLayer #LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
    θ::Function # library of NL transformations
    Ξactive::BitMatrix # active coefficients
end

# define how parameters are initialized
function LuxCore.initialparameters(rng::AbstractRNG, l::SINDyLayer)
    if typeof(l.init_weight) <: Function
        return (weight=l.init_weight(rng, l.out_dims, l.in_dims) .* l.Ξactive,
            bias=l.init_bias(rng, l.out_dims, 1))
    else
        return (weight=l.init_weight .* l.Ξactive,
            bias=l.init_bias)
    end
end

LuxCore.initialstates(::AbstractRNG, ::SINDyLayer) = NamedTuple()

LuxCore.parameterlength(l::SINDyLayer) = l.out_dims * l.in_dims

# Define the forward pass
function (l::SINDyLayer)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * l.θ(x)
    return y, st
end

function (l::SINDyLayer)(x::AbstractVector, ps, st::NamedTuple)
    y = ps.weight * l.θ(x)
    return vec(y), st
end

# constructors to build layers
function SINDyLayer(in_dims::Int, out_dims::Int, θ::Function; init_weight=glorot_uniform, init_bias=zeros32)
    return SINDyLayer{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight,
        init_bias, θ, BitArray(ones(out_dims, in_dims)))
end

function SINDyLayer(in_dims::Int, out_dims::Int, θ::Function, Ξactive::BitMatrix; init_weight=glorot_uniform, init_bias=zeros32)
    @assert size(Ξactive) == (out_dims, in_dims)
    #init_weight = zeros(Int32, out_dims, in_dims)
    return SINDyLayer{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight,
        init_bias, θ, Ξactive::BitMatrix)
end

function SINDyLayer(x, Lib, out_dim)
    #in_dim = size(x,1)
    in_dim = size(Lib(x), 1)
    return SINDyLayer(in_dim, out_dim, Lib)
end

function SINDyLayer(x, Lib, out_dim, Ξactive)
    #in_dim = size(x,1)
    in_dim = size(Lib(x), 1)
    return SINDyLayer(in_dim, out_dim, Lib, Ξactive)
end


# ======= SINDyNet ==========
# constructor for building network from layers
function SINDyNet(Libs, widths)
    @assert length(Libs) == length(widths) - 1
    L = []
    for i=1:length(widths)-1
        yi = Libs[i](ones(widths[i]))
        in_dim_i = size(yi, 1)
        l = SINDyLayer(in_dim_i, widths[i+1], Libs[i])
        push!(L, l)
    end

    return Chain(L..., name="SINDyNet")
end