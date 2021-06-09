# SDE.jl

"""
    abstract type AbstactSDE <: Any

subtype: SDE.

A supertype for implementing SDE types.
dX_t=θ(X_t)dt + σ(X_t)dW_t.
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
functions R→R. The variance is given by model.σ (either a function or a Float64).
Beginvalue X_0=model.beginvalue and model.Δ>0 is the precision with which we
discretise.

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
SDE(model::T, b::S) where {S<:Function, T<:SDEModel} = SDE(b, model)

function calculatenextsamplevalue(prevXval::Float64, b::Function,
    model::SDEModel{Float64}, BMincrement::Float64)
  return prevXval + b(prevXval)*model.Δ + model.σ*BMincrement
end

function calculatenextsamplevalue(prevXval::Float64, b::Function,
    model::T, BMincrement::Float64) where T<:SDEModel{S} where S<:Function
  return prevXval + b(prevXval)*model.Δ + model.σ(prevXval)*BMincrement
end


"""
  rand(sde::SDE)

Returns a SamplePath object which represents a sample path from an
AbstractSDE subtype sde. From time 0.0 to time sde.model.endtime, discretised
with precision sde.model.Δ.

# Examples

```julia
model = SDEModel(1.0,0.0,10.0,0.01)
sde = SDE(sin, model)
X = rand(sde)
```
"""
function rand(sde::SDE)::SamplePath
  timeinterval = 0.0:sde.model.Δ:sde.model.endtime
  lengthoftimeinterval = length(timeinterval)
  BMincrements = sqrt(sde.model.Δ) .* randn(lengthoftimeinterval)
  samplevalues = Array{Float64}(undef, lengthoftimeinterval)
  prevXval = samplevalues[1] = sde.model.beginvalue
  for k in 2:lengthoftimeinterval
    samplevalues[k] = calculatenextsamplevalue(prevXval, sde.b, sde.model,
        BMincrements[k])
    prevXval = samplevalues[k]
  end
  return SamplePath(timeinterval, samplevalues)
end