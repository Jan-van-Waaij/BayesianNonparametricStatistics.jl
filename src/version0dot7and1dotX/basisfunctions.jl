# basisfunctions.jl

"""
    fourier(k::Int)

fourier implements the Fourier basis of functions ϕ_k, defined by ϕ_0≡1, and if k≥1 and k is odd, by
ϕ_k(x)=sqrt(2)sin((k+1)*π*x) and if k≥1 and even, by ϕ_k(x)=sqrt(2)cos(π*k*x).
This function is defined for k≥0.

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
    return x -> (y = mod(x, 1.0); sqrttwo * sinpi((k+1)*y))
  else
    return x -> (y = mod(x, 1.0); sqrttwo * cospi(k*y))
  end
end

"""
    faberschauderone(x)

Implements the first Faber-Schauder function defined by 1-2x for 0≤x≤1/2 and
-1+2x for 0.5≤x≤1, and is 1-periodically extended to all x∈R.

#See also: faberschauder

#Warning

Note the difference between faberschauderone and faberschauder. The first is a 
function that takes a Float64 and returns a Float64, the second takes (j,k) and
returns an anonymous function that takes a Float64 and returns a Float64.

#Examples

```julia
x=-2.0:0.001:2.0
y=faberschauderone.(x)
```
"""
function faberschauderone(x)
    y = mod(x, 1.0)
    if 0≤y≤0.5
        return 1.0 - 2*y
    else
        return -1.0 + 2*y
    end
end

"""
    faberschauder(j::Int, k::Int)

faberschauder implements the k-th Faber-Schauder function of level j. Here, j≥0 and 1≤k≤2^j.
It is a one-periodic function and defined on [0,1] by 2^(j+1)(x-(k-1)2^(-j)) on
(k-1)2^(-j)≤x≤(k-1/2)2^(-j) and 1 - 2^(j+1)(x-(k-1/2)2^(-j)) on
[(k-1/2)2^(-j), k2^(-j)] and zero outside these intervals.

# See also: faberschauderone.

#Warning

Note the difference between faberschauderone and faberschauder. The first is a
function that takes a Float64 and returns a Float64, the second takes (j,k) and
returns an anonymous function that takes a Float64 and returns a Float64.

#Example with Plots
```julia
using Plots
J=2
x=0.0:0.001:1.0
p=plot()
for j in 0:J
  for k in 1:2^j
      y = faberschauder(j,k).(x)
      plot!(p,x,y)
  end
end
display(p)
```
"""
function faberschauder(j::Int,k::Int)
  j ≥ 0 || throw(AssertionError("j should be a nonnegative integer."))
  1≤k≤2^j || throw(AssertionError("k should be an integer between 1 and 2^j."))
  return function(x::Float64)
    y = mod(x,1.0)
    if y ≤ ldexp(float(k-1), -j) || y ≥ ldexp(float(k), -j)
      return 0.0
    elseif ldexp(float(k-1), -j) < y ≤ ldexp(k-0.5, -j)
      return ldexp(y, j+1)-2*(k-1)
    else
      return 2*k - ldexp(y, j+1)
    end
  end
end
