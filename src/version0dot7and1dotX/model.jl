# model.jl

"""
    abstract type AbstractModel end

subtype: SDEModel

AbstractModel is a supertype for statistical models.
"""
abstract type AbstractModel end

"""
    struct SDEModel{T<:Union{Float64, <:Function}} <: AbstractModel
        σ::T
        beginvalue::Float64
        endtime::Float64
        Δ::Float64
    end

# Constructors

```julia
    SDEModel(σ::Q, beginvalue::R, endtime::S, Δ::T) where {Q<:Real, R<:Real,
        S<:Real, T<:Real}
    SDEModel(σ::Q, beginvalue::R, endtime::S, Δ::T) where {Q<:Function, R<:Real,
        S<:Real, T<:Real}
```

SDEModel implements an SDE model with begin value X_0=beginvalue (at time zero) variance
σ, up to end time endtime. The SDE is discretised with precision Δ.

# Warning
It is assumed that for every b under consideration, the laws of dX_t=b(X_t)dt+σ(X_t)dW_t are
equivalent.

# Examples

```julia
SDEModel(1.0, 0.0, 10.0, 0.01)
# Implements the model dX_t=b(X_t)dt+σ(X_t)dW_t, with X_0=0.0, up to time 10,
# with precision 0.01.
#
SDEModel(identity, 0.0, 100.0, 0.1)
# Implements the model dX_t=b(X_t)+X_tdW_t, with X_0=0.0, up to time 100,
# discretised with precision 0.1.
```
"""
struct SDEModel{T} <: AbstractModel where T<:Union{Float64, <:Function}
    σ::T
    beginvalue::Float64
    endtime::Float64
    Δ::Float64

    function SDEModel(σ::Q, beginvalue::Float64, endtime::Float64, Δ::Float64) where {
            Q<:Union{Float64, <:Function}}
        endtime > 0 || throw(ArgumentError("Endtime should be positive."))
        Δ > 0 || throw(ArgumentError("Δ should be positive."))
        new{typeof(σ)}(σ, beginvalue, endtime, Δ)
    end
end
