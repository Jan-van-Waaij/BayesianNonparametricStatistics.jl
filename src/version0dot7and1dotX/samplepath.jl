# samplepath.jl

"""
    abstract type AbstractSamplePath <: Any

subtype: SamplePath.

Its subtype implements a continuous samplepath.
"""
abstract type AbstractSamplePath end

"""
    isstrictlyincreasing(x::AbstractVector{T})::Bool where T <: Number
    isstrictlyincreasing(x::AbstractRange{T})::Bool where T <: Number

Internal function, not exported.

Tests whether a vector of numbers is strictly increasing.
"""
function isstrictlyincreasing(x)
  for i in 1:length(x)-1
    x[i+1] <= x[i] && return false
  end
  return true
end

isstrictlyincreasing(x::AbstractRange) = step(x) > 0

"""
    SamplePath(timeinterval::S, samplevalues::T) where
        {S<:AbstractVector{Float64}, T<:AbstractVector{Float64}}

 SamplePath implements a continuous samplepath where the timeinterval is not necessarily
 equally spaced. Sample value samplevalues[k] is the value of the process at
 time timeinterval[k]. timeinterval is a strictly increasing vector. timeinterval and
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
    isstrictlyincreasing(timeinterval) ||
      throw(ArgumentError("Timeinterval should be increasing"))
    new{S,T}(timeinterval, samplevalues)
  end
end

#constructor
SamplePath(timeinterval, f::Function) = SamplePath(timeinterval, f.(timeinterval))

"""
    Base.step(X::SamplePath{<:AbstractRange})

Returns the step size of SamplePath.timeinterval.

# Example

```julia
t = 0.0:0.01:2.0
X = SamplePath(t, sinpi)
step(X) == step(t)
```
"""
Base.step(X::SamplePath{<:AbstractRange}) = Base.step(X.timeinterval)

"""
    Base.length(X::SamplePath)

Returns the length of the vector timeinterval == length vector samplevalues.

# Examples
```julia
X = SamplePath([0.,1.,2.], [3.,5., -1.])
length(X) # == 3
```

```julia
X = SamplePath(0.0:0.1:2??, sin)
length(X) == length(0.0:0.1:2??)
```
"""
Base.length(X::SamplePath)=length(X.timeinterval)

# For the functions below test need to be written.

"""
    Base.iterate(X::SamplePath, state=1)

Iterate over the sample values.

# Example
```julia
X = SamplePath([1.0, 2.0, 3.0], x->x^2)
for value in X
    println(value)
end
# Or in the reverse
for value in Iterators.Reverse(X)
    println(value)
end 
```  
"""
Base.iterate(X::SamplePath, state=1) = state > length(X) ? nothing : (X.samplevalues[state], state + 1)
Base.iterate(rX::Base.Iterators.Reverse{SamplePath{S,T}}, state=length(rX)) where {S, T} = state < 1 ? nothing : (rX.itr.samplevalues[state], state - 1) 

"""
    Base.eltype(::Type{SamplePath})

Outputs element type of SamplePath, which is Float64.
"""
Base.eltype(::Type{SamplePath{S,T}}) where {S,T} = Float64

"""
    Base.getindex(X::SamplePath, i::Int)

Returns the i-th observation of the samplepath, which is X.samplevalues[i].
"""
Base.getindex(X::SamplePath, i::Int) = X.samplevalues[i]

"""
    Base.firstindex(X::SamplePath)

The first index of the SamplePath is 1.
"""
Base.firstindex(X::SamplePath) = 1

"""
    Base.lastindex(X::SamplePath)

The last index of the SamplePath is length(X).
"""
Base.lastindex(X::SamplePath) = length(X)

