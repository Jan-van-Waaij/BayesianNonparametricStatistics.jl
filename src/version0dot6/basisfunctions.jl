# basisfunctions.jl

"""
    fourier(k::Int)

Implements the Fourier basis of functions ϕ_k, defined, if k is odd, by
ϕ_k(x)=sqrt(2)sin((k+1)π*x) and if k is even, by ϕ_k(x)=sqrt(2)cos(k*x).
This function is defined for k≥0 and ϕ_0≡1. 

# Examples

```julia 
x = -1.0:0.01:2.0
y = fourier(3).(x)
```
"""
function fourier(k::Int)
  k ≥ 0 || throw(AssertionError("k should be non-negative"))
  sqrttwo = sqrt(2.0)
  if k == 0
    return x -> 1.0
  elseif isodd(k)
    return x -> sqrttwo * sinpi(float(k+1)*x)
  else
    return x -> sqrttwo * cospi(float(k)*x)
  end
end

"""
    faberschauderone(x::Float64)

Implements the first Faber-Schauder function defined by 1-2x for 0≤x≤1/2 and
-1+2x for 0.5≤x≤1, and is 1-periodically extended to all x∈R.

#See also: faberschauder

#Warning

Note the difference between faberschauderone and faberschauder. The first is a
function that takes a Float64 and returns a Float64, the second takes (j,k) and
returns a anonymous function that takes a Float64 and returns a Float64.

#Examples

```julia
x=-2.0:0.001:2.0
y=faberschauderone.(x)
```
"""
function faberschauderone(x::Float64)
    y = mod(x, 1.0)
    if 0≤y≤0.5
        return 1.0 - 2*y
    else
        return -1.0 + 2*y
    end
end

"""
    faberschauder(j::Int, k::Int)

Implements the k-th Faber-Schauder function of level j. Here, j≥0 and 1≤k≤2^j.
It is a one periodic function and defined on [0,1] by 2^(j+1)(x-(k-1)2^(-j)) on
(k-1)2^(-j)≤x≤(k-1/2)2^(-j) and 1 - 2^(j+1)(x-(k-1/2)2^(-j)) on
[(k-1/2)2^(-j), k2^(-j)] and zero outside these intervals.

# See also: faberschauderone.

#Warning

Note the difference between faberschauderone and faberschauder. The first is a
function that takes a Float64 and returns a Float64, the second takes (j,k) and
returns a anonymous function that takes a Float64 and returns a Float64.

#Example using PyPlot
```julia
using PyPlot
clf()
j=3
x=-2.0:0.001:2.0
for k in 1:2^j
    y = faberschauder(j,k).(x)
    plot(x,y)
end
```
"""
function faberschauder(j::Int,k::Int)
  j ≥ 0 || throw(AssertionError("j should be a nonnegative integer."))
  1≤k≤2^j || throw(AssertionError("k should be an integer between 1 and 2^j."))
  twotothepowerminusj = 2.0^-j
  k_float = float(k)
  j_float = float(j)
  return function(x::Float64)
    y = mod(x,1.0)
    if y ≤ (k_float-1.0)*twotothepowerminusj || y ≥ k_float*twotothepowerminusj
      return 0.0
    elseif (k_float-1.0)*twotothepowerminusj < y ≤ (k_float-0.5)*twotothepowerminusj
      return 2.0^(j_float+1.0)*y-2*k_float+2.0
    else
      return 2.0*k_float - 2.0^(j_float+1.0) * y
    end
  end
end
