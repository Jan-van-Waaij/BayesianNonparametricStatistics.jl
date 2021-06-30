# SDE.jl

"""
    abstract type AbstactSDE <: Any

subtype: SDE.

A supertype for implementing SDEs.
"""
abstract type AbstractSDE end

"""
    struct SDE{S, T} <: AbstractSDE where {S<:Function, T<:SDEModel}
      b::S
      model::T
    end

# Constructors
```julia
SDE(b::S, model::T) where {S<:Function, T<:SDEModel}
SDE(model::T, b::S) where {S<:Function, T<:SDEModel}
```

Implements a stochastic differential equation dX_t=b(X_t)dt+σ(X_t)dW_t on time
interval [0,model.endtime], with W_t a Brownian motion. The drift function b is
a function on the real line. model.σ (either a function or a number) gives the diffusion.
The begin value X_0=model.beginvalue and the sample path is discretised via the
Euler-Maruyama scheme with time steps model.Δ.

# Example

```julia
model = SDEModel(x->2+sin(x), 0.0,10.0,0.001)
sde = SDE(sin, model)
```
"""
struct SDE{S, T} <: AbstractSDE where {S<:Function, T<:SDEModel}
  b::S
  model::T
end

# Constructor
SDE(model::SDEModel, b::Function) = SDE(b, model)

"""
    calculatenextsamplevalue(prevXval, σ::Float64, b, Δ, BMincrement)
    calculatenextsamplevalue(prevXval, σ, b, Δ, BMincrement)


Internal function, not exported.

Given the previous samplepath value prevXval, it calculates the next sample value with the Euler-Maruyama scheme. 
"""
calculatenextsamplevalue(prevXval, σ::Float64, b, Δ, BMincrement) = prevXval + b(prevXval)*Δ + σ*BMincrement
calculatenextsamplevalue(prevXval, σ, b, Δ, BMincrement) = prevXval + b(prevXval)*Δ + σ(prevXval)*BMincrement


"""
  rand(sde::SDE)

Returns a sample path from the SDE sde, from time 0.0 to time sde.model.endtime, 
discretised with precision sde.model.Δ with the help of the Euler-Maruyama scheme.

# Examples

```julia
model = SDEModel(1.0,0.0,10.0,0.01)
sde = SDE(sin, model)
X = rand(sde)
```
"""
function rand(sde::SDE)
    timeinterval = 0.0:sde.model.Δ:sde.model.endtime
    lengthoftimeinterval = length(timeinterval)
    BMincrements = sqrt(sde.model.Δ) .* randn(lengthoftimeinterval)
    samplevalues = Array{Float64}(undef, lengthoftimeinterval)
    prevXval = samplevalues[1] = sde.model.beginvalue
    for k in 2:lengthoftimeinterval
        prevXval = samplevalues[k] = calculatenextsamplevalue(prevXval, sde.model.σ, sde.b, sde.model.Δ, BMincrements[k])
    end
    return SamplePath(timeinterval, samplevalues)
end