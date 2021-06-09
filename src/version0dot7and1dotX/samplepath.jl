# samplepath.jl

"""
    abstract type AbstractSamplePath <: Any

subtype: SamplePath.

Its subtype implements a continuous samplepath.
"""
abstract type AbstractSamplePath end

"""
    isincreasing(x::AbstractVector{T})::Bool where T <: Number
    isincreasing(x::AbstractRange{T})::Bool where T <: Number

Tests whether a vector of numbers is strictly increasing. Is internal to
NonparametricBayesForDiffusions.
"""
function isincreasing(x::S) where S<:AbstractArray{T} where T <: Number
  for i in 1:length(x)-1
    x[i+1] <= x[i] && return false
  end
  return true
end

isincreasing(x::S) where S<:AbstractRange{T} where T <: Number = step(x) > 0

"""
    SamplePath(timeinterval::S, samplevalues::T) where
        {S<:AbstractVector{Float64}, T<:AbstractVector{Float64}}

 Implements a continuous samplepath where the timeinterval is not necessarily
 equally spaced. Sample value samplevalues[k] is the value of the process at
 time timeinterval[k]. timeinterval is an increasing vector. timeinterval and
 samplevalues are of equal length.

 # Performance hint: 

 Use AbstractRange objects for SamplePath.timeinterval whenever possible.

# Constructors:

```julia
    SamplePath(timeinterval, samplevalues)
    SamplePath(timeinterval, f::Function) =
        SamplePath(timeinterval, f.(timeinterval))
```

# Examples

```julia
t = 0.0:0.01:2.0
X = SamplePath(t, sinpi)
```

```julia
t = 0.0:0.01:1.0
v = map(x->floor(10*x), t)
X = SamplePath(t, v)
```
"""
struct SamplePath{S, T} <: AbstractSamplePath where {S<:AbstractVector{Float64},
        T<:AbstractVector{Float64}}
  timeinterval::S
  samplevalues::T

  function SamplePath(timeinterval::S, samplevalues::T) where {
      S<:AbstractVector{Float64}, T<:AbstractVector{Float64}}
    length(timeinterval) == length(samplevalues) ||
      throw(DimensionMismatch("Length of timeinterval should be equal to the length of the
        samplevalues vector"))
    isincreasing(timeinterval) ||
      throw(ArgumentError("Timeinterval should be increasing"))
    new{S,T}(timeinterval, samplevalues)
  end
end

#constructor
SamplePath(timeinterval, f::Function) = SamplePath(timeinterval, f.(timeinterval))

"""
    Base.step(X::SamplePath{S}) where S<:AbstractRange{T} where T<:Number

Returns the step size of SamplePath.timeinterval.

# Example

```julia
t = 0.0:0.01:2.0
X = SamplePath(t, sinpi)
step(X) == step(t)
```
"""
Base.step(X::SamplePath{S}) where S<:AbstractRange{T} where T<:Number = Base.step(X.timeinterval)

"""
    Base.length(X::SamplePath)

Returns the length of the vector timeinterval == length vector samplevalues.

# Examples
```julia
X = SamplePath([0.,1.,2.], [3.,5., -1.])
length(X)
```

```julia
X = SamplePath(0.0:0.1:2π, sin)
length(X) == length(0.0:0.1:2π)
```
"""
Base.length(X::SamplePath)=length(X.timeinterval)

"""
    Base.iterate(X::SamplePath, state=1)

Iterate over the sample values.     
"""
Base.iterate(X::SamplePath, state=1) = state > length(X) ? nothing : (X.samplevalues[state], state + 1)

"""
    Base.eltype(::Type{SamplePath})

Outputs element type of SamplePath, which is Float64.
"""
Base.eltype(::Type{SamplePath}) = Float64

"""
    Base.getindex(X::SamplePath, i::Int)

Returns the i-th observation of the samplepath, that is X.samplevalues[i].
"""
function Base.getindex(X::SamplePath, i::Int) = X.samplevalues[i]

"""
    Base.firstindex(X::SamplePath)

The first index of SamplePath is 1.
"""
Base.firstindex(X::SamplePath) = 1

"""
    Base.lastindex(X::SamplePath)

The last index of SamplePath is length(X).
"""
Base.lastindex(X::SamplePath) = length(X)


